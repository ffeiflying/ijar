import sys
import os
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # IJAR
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # cf_heatmap

from envidence import EvidenceModel
from diff_cf import differentiable_kappa


def reference_kappa(X, B):
    """逐 token 循环现有（不可微）EvidenceModel 作为参考值。"""
    ev = EvidenceModel(B.detach())
    out = []
    for t in range(X.shape[1]):
        ev.get_evidence_weights(X[:, t:t + 1].detach())
        out.append(ev.get_evidence_conflict().item())
    return np.array(out)


def test_equivalence():
    torch.manual_seed(0)
    J, K, T = 64, 128, 20
    B = torch.randn(J, K)
    X = torch.randn(K, T)
    got = differentiable_kappa(X, B).detach().numpy()
    ref = reference_kappa(X, B)
    rel = np.abs(got - ref) / (np.abs(ref) + 1e-9)
    assert np.all(rel < 1e-2), f"max rel diff {rel.max():.2e} (got {got[:3]}, ref {ref[:3]})"
    print(f"equivalence OK  max rel diff = {rel.max():.2e}")


def test_shape():
    torch.manual_seed(1)
    B = torch.randn(32, 64)
    X = torch.randn(64, 7)
    k = differentiable_kappa(X, B)
    assert tuple(k.shape) == (7,)
    print("shape OK", tuple(k.shape))


def test_gradient():
    torch.manual_seed(2)
    B = torch.randn(32, 64)
    X = torch.randn(64, 5, requires_grad=True)
    s = differentiable_kappa(X, B).sum()
    s.backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
    print(f"gradient OK  mean|grad| = {X.grad.abs().mean().item():.4f}")


if __name__ == "__main__":
    test_equivalence()
    test_shape()
    test_gradient()
    print("ALL PASS")
