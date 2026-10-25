import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gradcam import gradcam_saliency


def test_gradcam():
    torch.manual_seed(0)
    S, C = 12, 5
    features = torch.randn(S, C)
    grads = torch.randn(S, C)
    cam = gradcam_saliency(features, grads)
    assert tuple(cam.shape) == (S,)
    assert (cam >= 0).all()
    alpha = grads.mean(dim=0)
    ref = torch.relu((features * alpha).sum(dim=1))
    assert torch.allclose(cam, ref)
    print("gradcam OK")


if __name__ == "__main__":
    test_gradcam()
    print("ALL PASS")
