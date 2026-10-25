"""
Equivalence test: closed-form ACF (acf_core.acf_from_phi) vs the reference
implementation (IJAR/envidence.py, materialised (J, K) tensor).

This isolates the algebra from the model, so a failure here means the closed
form is wrong -- not that the hooks are wrong.

Run:
    /data/liuxiaofei/miniconda3/envs/EUQ/bin/python test_acf_equivalence.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_IJAR_DIR = os.path.dirname(_HERE)
for p in (_HERE, _IJAR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from envidence import EvidenceModel                      # noqa: E402
from acf_core import acf_from_phi, precompute_layer_stats  # noqa: E402


def reference(B, phi):
    """Per-column kappa/ignorance via envidence.py, exactly as run_qwen.py does."""
    kappas, igns = [], []
    for t in range(phi.shape[1]):
        em = EvidenceModel(B.clone())
        em.get_evidence_weights(phi[:, t].unsqueeze(1))  # (K, 1) == phi_t.T
        kappas.append(em.get_evidence_conflict().item())
        igns.append(em.get_evidence_ignorance().item())
    return torch.tensor(kappas, dtype=torch.float64), torch.tensor(igns, dtype=torch.float64)


def check(name, B, phi, rtol=1e-6):
    m, _, _ = precompute_layer_stats(B)
    k_fast, i_fast = acf_from_phi(B, m, phi)
    k_ref, i_ref = reference(B, phi)
    k_ref = k_ref.to(k_fast.device)
    i_ref = i_ref.to(i_fast.device)

    dk = (k_fast - k_ref).abs().max().item()
    di = (i_fast - i_ref).abs().max().item()
    rk = dk / max(k_ref.abs().max().item(), 1e-30)
    ri = di / max(i_ref.abs().max().item(), 1e-30)
    ok = rk < rtol and ri < rtol
    print(
        f"  {name:34s} d_kappa={dk:.3e} (rel {rk:.2e})  "
        f"d_ign={di:.3e} (rel {ri:.2e})  {'OK' if ok else 'FAIL'}"
    )
    if not ok:
        print(f"    ref  kappa[:4] = {k_ref[:4].tolist()}")
        print(f"    fast kappa[:4] = {k_fast[:4].tolist()}")
    return ok


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    all_ok = True

    print("closed-form vs envidence.py\n" + "-" * 68)

    # small, dense
    J, K, T = 32, 64, 5
    B = torch.randn(J, K, device=dev)
    phi = torch.randn(K, T, device=dev)
    all_ok &= check("randn float32", B, phi)

    # weights/activations shaped like a real down_proj (small magnitudes, both signs)
    B = (torch.randn(J, K, device=dev) * 0.02)
    phi = (torch.randn(K, T, device=dev) * 3.0)
    all_ok &= check("small-weight float32", B, phi)

    # all-negative -> P == 0 for every token (reference would IndexError here)
    B = -torch.rand(J, K, device=dev) - 0.1
    phi = torch.rand(K, T, device=dev)
    m, _, _ = precompute_layer_stats(B)
    k_fast, i_fast = acf_from_phi(B, m, phi)
    finite = bool(torch.isfinite(k_fast).all() and torch.isfinite(i_fast).all())
    print(f"  {'all-negative (P==0)':34s} kappa={k_fast[0].item():.3e}  "
          f"{'OK (degenerate, finite)' if finite else 'FAIL'}")
    all_ok &= finite

    # bigger, closer to the 32B geometry
    J, K, T = 512, 1024, 8
    B = torch.randn(J, K, device=dev) * 0.01
    phi = torch.randn(K, T, device=dev)
    all_ok &= check("wide float32", B, phi)

    # float16 (what the runner actually feeds in)
    if dev == "cuda":
        J, K, T = 64, 128, 6
        B = (torch.randn(J, K, device=dev) * 0.02).half()
        phi = (torch.randn(K, T, device=dev) * 3.0).half()
        all_ok &= check("small-weight float16", B, phi, rtol=1e-4)

    print("-" * 68)
    print("ALL OK" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
