"""
Core ACF computation for the model-scaling experiment (see DESIGN.md).

Two pieces:

1. `DownProjCapture` — grab the last-position input to every text decoder
   layer's `mlp.down_proj` while `model.generate()` runs, using forward
   pre-hooks.  No modification to the conda env's transformers.

2. `acf_from_phi` — the DST evidence conflict (kappa) / ignorance, numerically
   equivalent to `IJAR/envidence.py` + `layerauroc/run_qwen.py`, but in closed
   form so it stays feasible for a 32B model.

   The only mathematical liberty is collapsing the (J, K) tensor that
   `EvidenceModel.get_evidence_weights` materialises: every column of that
   tensor is identical and only its row sums are consumed downstream, so the
   whole thing reduces to O(J) per token.  See DESIGN.md for the derivation.
"""

import torch


# --------------------------------------------------------------------------
# Text decoder layer discovery
# --------------------------------------------------------------------------

_LAYER_PATHS = (
    "model.language_model.layers",  # Qwen3-VL
    "model.layers",                 # Qwen2.5-VL
    "language_model.layers",
    "layers",
)


def resolve_text_layers(model):
    """Return the ModuleList of text decoder layers, whatever the wrapper."""
    for path in _LAYER_PATHS:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is None or not hasattr(obj, "__len__") or len(obj) == 0:
            continue
        first = obj[0]
        if hasattr(first, "mlp") and hasattr(first.mlp, "down_proj"):
            return obj
    raise RuntimeError(
        "Could not locate text decoder layers (tried: %s)" % ", ".join(_LAYER_PATHS)
    )


# --------------------------------------------------------------------------
# Down-proj input capture
# --------------------------------------------------------------------------

class DownProjCapture:
    """
    Context manager.  Inside the `with` block, every forward call that reaches
    any layer's `down_proj` appends the last-position input vector to
    `self.buf[layer_idx]`.

    `model.generate()` calls forward once for prefill and once per decoded
    token, so each buffer ends up with `1 + n_new_tokens` entries -- the same
    count and the same content as the patched-transformers path in
    `layerauroc/run_qwen.py`, which took `layer_outputs[-1][:, -1:, :]`.

    Usage:
        with DownProjCapture(model) as cap:
            out = model.generate(...)
        # cap.buf[i] -> list of (batch, hidden) tensors, one per generation step
    """

    def __init__(self, model):
        self.layers = resolve_text_layers(model)
        self.n_layers = len(self.layers)
        self.buf = None
        self._handles = []

    def _make_hook(self, idx):
        def hook(module, args):
            x = args[0]
            # (batch, seq, intermediate) -> last position, kept on device
            self.buf[idx].append(x[:, -1, :].detach())
        return hook

    def __enter__(self):
        self.buf = [[] for _ in range(self.n_layers)]
        for i, layer in enumerate(self.layers):
            h = layer.mlp.down_proj.register_forward_pre_hook(self._make_hook(i))
            self._handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    # -- convenience ------------------------------------------------------
    def phi_per_layer(self):
        """
        Stack the captured steps into a (K, T) float tensor per layer.

        Returns list of length n_layers; entry i is (intermediate, T) where
        T = number of generation steps (prefill + generated tokens).
        """
        out = []
        for steps in self.buf:
            if not steps:
                out.append(None)
                continue
            # each step is (1, K) for batch size 1
            stacked = torch.cat([s.reshape(-1) for s in steps], dim=0)  # (T*K,)
            out.append(stacked.reshape(len(steps), -1).t().contiguous())  # (K, T)
        return out


# --------------------------------------------------------------------------
# Closed-form ACF
# --------------------------------------------------------------------------

def precompute_layer_stats(weight):
    """
    weight : down_proj.weight, shape (J, K) = (hidden, intermediate)

    Returns (m, J, Ki) where m = weight.mean(dim=1), the row mean that the
    reference implementation subtracts before the matvec
    (`B_star = B - B.mean(axis=1, keepdims=True)`).
    """
    J, Ki = weight.shape
    m = weight.mean(dim=1)
    return m, J, Ki


def acf_from_phi(weight, m, phi):
    """
    Compute per-token conflict kappa and ignorance.

    weight : (J, K) down_proj weight (any float dtype, on any device)
    m      : (J,)  weight.mean(dim=1), precomputed by `precompute_layer_stats`
    phi    : (K, T) captured down_proj inputs, one column per generation step

    Returns (kappa, ignorance), each a float64 tensor of shape (T,).

    Mirrors `layerauroc/run_qwen.py`: the centring and the matvec both run in
    the weight's dtype, everything after is float64.  Centring *before* the
    matvec (rather than subtracting `m * phi.sum()` afterwards) is what keeps
    the fp16 rounding in line with the reference implementation.
    """
    J, Ki = weight.shape
    phi = phi.to(device=weight.device, dtype=weight.dtype)

    # w[j, t] = (B_star[j] . phi[:, t]) / J
    b_star = weight - m[:, None]        # (J, K); temp is freed right after the matvec
    w = (b_star @ phi) / J              # (J, T)
    del b_star
    w = w.to(torch.float64)

    # Every column of the (J, K) evidence-weight tensor equals w[:, t], so
    # `w_pos1` (shape (T, K) in the reference) is K copies of a single value
    # per token -> collapse to that scalar.  The factor K then reappears each
    # time the reference sums over that K axis.
    w_pos1 = torch.relu(w).sum(dim=0)       # P[t] = sum_j relu(w[j, t])
    w_neg1 = torch.relu(-w).sum(dim=0)      # N[t] = sum_j relu(-w[j, t])

    # eta_pos_temp = 1 / (sum_k e^{P} - K + 1)
    eta_pos = 1.0 / (Ki * torch.exp(w_pos1) - Ki + 1.0)
    # eta_neg_temp = 1 / (1 - prod_k (1 - e^{-N}))
    eta_neg = 1.0 / (1.0 - torch.pow(1.0 - torch.exp(-w_neg1), Ki))

    # The reference implementation replaces zeros with the smallest strictly
    # positive entry.  Since all K entries of a row are identical, that is a
    # no-op whenever the value is positive.
    w_pos2 = w_pos1          # P' == P for P > 0 (degenerate P == 0 -> kappa 0)
    w_neg2 = torch.where(
        w_neg1 > 0, w_neg1, torch.full_like(w_neg1, 1e-6)
    )                        # N' == N for N > 0, else 1e-6 as in the original

    kappa = Ki * eta_pos * (torch.exp(w_pos2) - 1.0) * (1.0 - eta_neg * torch.exp(-w_neg1))
    eta_temp = 1.0 / (1.0 - kappa)
    ignorance = Ki * torch.exp(-w_neg2) * eta_temp * eta_pos * eta_neg

    return kappa, ignorance


def layer_acf(weight, m, phi):
    """Mean conflict / ignorance over generation steps for one layer."""
    kappa, ignorance = acf_from_phi(weight, m, phi)
    return float(kappa.mean().item()), float(ignorance.mean().item())
