"""Reference Denoeux2019 GLR decomposition, equations25/27/38 and AppendixA.
Only meaningful as classifier interpretation when weight rows are class logits.
Not an automatic modality-conflict score for arbitrary hidden-channel projections.
"""
import numpy as np

def logsumexp(v):
    m=np.max(v)
    return m+np.log(np.exp(v-m).sum()) if np.isfinite(m) else m

def evidence(weight,phi,bias=None,mu=None):
    """PyTorch weight[K,J], phi[J], fixed calibration mean mu[J]."""
    weight=np.asarray(weight,dtype=np.float64);phi=np.asarray(phi,dtype=np.float64)
    K,J=weight.shape
    assert K>=2 and phi.shape==(J,)
    beta=weight.T
    beta=beta-beta.mean(axis=1,keepdims=True)
    b=np.zeros(K) if bias is None else np.asarray(bias,dtype=np.float64)
    b=b-b.mean()
    mu=np.zeros(J) if mu is None else np.asarray(mu,dtype=np.float64)
    alpha=(b+mu@beta)[None,:]/J-beta*mu[:,None]
    w=beta*phi[:,None]+alpha
    assert w.shape==(J,K)
    return w

def summarize(w):
    """Stable cross-stage conflict, ignorance and plausibility from valid w[J,K]."""
    w=np.asarray(w,dtype=np.float64)
    a=np.maximum(w,0).sum(axis=0);d=np.maximum(-w,0).sum(axis=0)
    with np.errstate(divide='ignore',invalid='ignore'):
        lexp=np.where(a>0,a+np.log(-np.expm1(-a)),-np.inf)
        lprod=np.log(-np.expm1(-d)).sum()
        ldm=np.log(-np.expm1(lprod))
    ldp=logsumexp(np.r_[0.,lexp])
    lz=logsumexp(np.r_[ldm,lexp-d])
    log_nonconflict=lz-ldp-ldm
    conflict=-np.expm1(min(log_nonconflict,0))
    ignorance=np.exp(-d.sum()-lz)
    lpl=a-d-lz
    probs=np.exp(lpl-logsumexp(lpl))
    return {'conflict':float(conflict),'ignorance':float(ignorance),'normalized_plausibility':probs,'w_plus':a,'w_minus':d}
