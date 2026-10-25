"""GradCAM 空间显著性（标准公式）。"""
import torch


def gradcam_saliency(features, grads):
    """GradCAM-style spatial saliency.

    Args:
        features: tensor [..., S, C] — target-layer activations (C channels at
            each of S spatial positions; leading dims broadcast, e.g. a batch).
        grads: tensor [..., S, C] — gradients of the scored scalar wrt features.

    Returns:
        cam: tensor [..., S] — non-negative per-position saliency.
    """
    lead, C = features.shape[:-1], features.shape[-1]
    f = features.reshape(-1, C)
    g = grads.reshape(-1, C)
    alpha = g.mean(dim=0)              # [C]
    cam = (f * alpha).sum(dim=1)       # [prod(lead)]
    return torch.relu(cam).reshape(*lead).float()   # 与 kappa 一致统一 fp32
