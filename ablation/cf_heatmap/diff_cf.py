"""可微闭式 kappa（DST 冲突因子）——与 envidence.EvidenceModel 数值等价但可反传。

参考 envidence.py 的 _calculate_basic_terms：对单 token 的 down-proj 输入 x：
  w = B̃ᵀx / J（B̃ = B - mean(B, dim=1)）
  u = Σⱼ relu(w)ⱼ,  v = Σⱼ relu(-w)ⱼ
  eta_pos = 1/(K*(e^u - 1) + 1)
  eta_neg = 1/(1 - (1 - e^-v)^K)
  kappa   = K * eta_pos * (e^u - 1) * (1 - eta_neg * e^-v)
用 expm1/log1p 保证 fp32 数值稳定。
"""
import torch


def differentiable_kappa(X, B):
    """Per-token DST conflict factor (kappa) from MLP down-proj inputs.

    Args:
        X: tensor [K, T] — T tokens, each a K-dim down-proj input
           (K = LLM intermediate_size).
        B: tensor [J, K] — pretrained down-proj weight (fixed, no grad needed).

    Returns:
        kappa: tensor [T] — per-token conflict factor (float32, differentiable).
    """
    J, K = B.shape
    X = X.float()                                    # bf16/fp16 模型统一到 fp32
    Bs = B - B.mean(dim=1, keepdim=True)            # [J, K]
    W = Bs @ X / J                                   # [J, T]
    u = torch.relu(W).sum(dim=0)                     # [T]
    v = torch.relu(-W).sum(dim=0)                    # [T]
    Kf = float(K)
    expm1u = torch.expm1(u)                          # [T]
    eta_pos = 1.0 / (Kf * expm1u + 1.0)              # [T]
    # eta_neg = 1/(1 - (1-e^-v)^K)；用 expm1/log1p 避免 fp32 下溢为 1
    eta_neg = -1.0 / torch.expm1(Kf * torch.log1p(-torch.exp(-v)))  # [T]
    kappa = Kf * eta_pos * expm1u * (1.0 - eta_neg * torch.exp(-v))  # [T]
    return kappa
