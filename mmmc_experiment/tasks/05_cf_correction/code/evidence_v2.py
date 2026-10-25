"""ACF evidence v2: latent-hypothesis DST, with explicit feature/output axes.

Preserves attention/FFN-level hypotheses; does not use LM logits or A/B prompts.
PyTorch weight is [hypotheses K, features J]. Feature means must come from a
fixed calibration set. With no mean supplied, alpha=0 (zero-mean gauge), not an
assertion that real activations are centered. All input features remain separate
until positive/negative evidence has been accumulated. Original envidence.py is
retained unchanged and is not imported here.
"""
import torch

class EvidenceModel:
    VERSION='acf_evidence_v2'
    def __init__(self,Beta,bias=None,feature_mean=None,temperature=1.0):
        if not isinstance(Beta,torch.Tensor) or Beta.ndim!=2:
            raise ValueError('Beta must be a torch weight[K hypotheses,J features]')
        if Beta.shape[0]<2 or temperature<=0:
            raise ValueError('Need >=2 hypotheses and positive temperature')
        self.B=Beta
        self.temperature=float(temperature)
        dtype=torch.float64 if Beta.dtype==torch.float64 else torch.float32
        b=Beta.detach().to(dtype=dtype)
        self.beta=b-b.mean(0,keepdim=True) # each feature centered across hypotheses
        self.feature_mean=torch.zeros(b.shape[1],dtype=dtype,device=b.device) if feature_mean is None else torch.as_tensor(feature_mean,dtype=dtype,device=b.device).reshape(-1)
        if self.feature_mean.numel()!=b.shape[1]:raise ValueError('Feature mean dimension mismatch')
        offset=torch.zeros(b.shape[0],dtype=dtype,device=b.device) if bias is None else torch.as_tensor(bias,dtype=dtype,device=b.device).reshape(-1)
        if offset.numel()!=b.shape[0]:raise ValueError('Bias dimension mismatch')
        self.bias_centered=offset-offset.mean()
        self.offset=(self.bias_centered+self.beta@self.feature_mean)/b.shape[1]
        self.evidence_weights=None

    def _construct_weights(self,Phi):
        phi=torch.as_tensor(Phi,device=self.beta.device,dtype=self.beta.dtype)
        if phi.ndim==2 and phi.shape[1]==1:phi=phi[:,0]
        if phi.ndim!=1 or phi.numel()!=self.beta.shape[1]:
            raise ValueError('Phi must be [J] or [J,1] for a single token')
        return (self.beta*(phi-self.feature_mean)[None,:]+self.offset[:,None])/self.temperature

    @torch.no_grad()
    def get_evidence_totals(self,Phi):
        """Same evidence, totals only; permits batched DST fusion after extraction."""
        w=self._construct_weights(Phi)
        return w.relu().sum(1,dtype=torch.float64),(-w).relu().sum(1,dtype=torch.float64)

    @torch.no_grad()
    def get_evidence_weights(self,Phi):
        w=self._construct_weights(Phi)
        self.evidence_weights=w.T.unsqueeze(0)
        self._calculate_basic_terms(w)
        return self.evidence_weights

    def _calculate_basic_terms(self,w):
        a=w.relu().sum(1,dtype=torch.float64)
        d=(-w).relu().sum(1,dtype=torch.float64)
        self.w_pos1=a[None,:];self.w_neg1=d[None,:];self.K=len(a)
        stats=combine_evidence(a,d)
        self.log_nonconflict=stats['log_nonconflict']
        self.kappa=stats['cf'].reshape(1)
        self.log_ignorance=stats['log_ignorance']
        self.ignorance=stats['ignorance'].reshape(1)
        self.log_plausibility=stats['log_plausibility']
        self.normalized_plausibility=stats['normalized_plausibility']
        self.log_normalizer=stats['log_normalizer']

    def get_evidence_conflict(self):return self.kappa
    def get_evidence_ignorance(self):return self.ignorance
    def get_conflict_logscore(self):
        """Monotone -log(1-CF) ranking when CF rounds to1; this is not raw CF."""
        return -self.log_nonconflict.reshape(1)
    def get_normalized_plausibility(self):return self.normalized_plausibility


def combine_evidence(w_plus,w_minus,temperature=1.0):
    """Public stable DST fusion; accepts [...,K] nonnegative totals.
    temperature is an explicit evidence scale, not a label-based sign change.
    """
    if temperature<=0:raise ValueError('temperature must be positive')
    a=torch.as_tensor(w_plus).to(dtype=torch.float64)/temperature
    d=torch.as_tensor(w_minus,device=a.device).to(dtype=torch.float64)/temperature
    if a.shape!=d.shape or a.shape[-1]<2:raise ValueError('Expected matching [...,K>=2] totals')
    if not torch.isfinite(a).all() or not torch.isfinite(d).all() or (a<0).any() or (d<0).any():raise ValueError('Evidence totals must be finite and nonnegative')
    log_expm1=a+torch.log(-torch.expm1(-a))
    zero=torch.zeros_like(a[...,:1])
    log_dplus=torch.logsumexp(torch.cat([zero,log_expm1],dim=-1),dim=-1)
    log_dminus=torch.log(-torch.expm1(torch.log(-torch.expm1(-d)).sum(-1)))
    log_z=torch.logsumexp(torch.cat([log_dminus[...,None],log_expm1-d],dim=-1),dim=-1)
    log_nc=(log_z-log_dplus-log_dminus).clamp(max=0)
    if not torch.isfinite(log_nc).all():raise FloatingPointError('Nonfinite CF')
    log_ig=-d.sum(-1)-log_z
    log_pl=a-d-log_z[...,None]
    return {'cf':(-torch.expm1(log_nc)).clamp(0,1),'log_nonconflict':log_nc,'cf_logscore':-log_nc,'log_ignorance':log_ig,'ignorance':log_ig.exp(),'log_plausibility':log_pl,'normalized_plausibility':torch.softmax(log_pl,dim=-1),'log_normalizer':log_z}


def binary_channel_evidence(w_plus,w_minus,temperature=1.0):
    """Separate binary polarity frame per latent channel (no mutual exclusivity across channels).
    Positive/negative feature contributions support opposite signs of that channel's
    relative projected response. Pooling across channels is a detection heuristic.
    """
    if temperature<=0:raise ValueError('temperature must be positive')
    a=torch.as_tensor(w_plus).to(torch.float64)/temperature
    d=torch.as_tensor(w_minus,device=a.device).to(torch.float64)/temperature
    if a.shape!=d.shape or (a<0).any() or (d<0).any():raise ValueError('Need nonnegative matching evidence')
    base=torch.logaddexp(-a,-d)
    log_nc=base+torch.log1p(-torch.exp(-a-d-base))
    return {'cf':(-torch.expm1(-a))*(-torch.expm1(-d)),'cf_logscore':-log_nc,'log_ignorance':-a-d-log_nc}


def cross_source_conflict(visual_plus,visual_minus,text_plus,text_minus,temperature=1.0):
    """Exact binary-frame DS conflict between visual and text/context masses.
    Each channel has its own common +/- frame. Sources can individually have
    internal conflict; each is normalized before inter-source fusion.
    """
    def log_masses(a,d):
        a=torch.as_tensor(a).to(torch.float64)/temperature
        d=torch.as_tensor(d,device=a.device).to(torch.float64)/temperature
        if a.shape!=d.shape or (a<0).any() or (d<0).any():raise ValueError('Invalid source evidence')
        base=torch.logaddexp(-a,-d)
        nc=base+torch.log1p(-torch.exp(-a-d-base))
        return torch.stack([torch.log(-torch.expm1(-a))-d-nc,torch.log(-torch.expm1(-d))-a-nc,-a-d-nc],dim=-1)
    if temperature<=0:raise ValueError('temperature must be positive')
    v=log_masses(visual_plus,visual_minus);t=log_masses(text_plus,text_minus)
    joints=v[..., :,None]+t[...,None,:]
    # Empty intersection occurs only for opposite singleton polarities.
    cross=torch.logsumexp(torch.stack([joints[...,0,1],joints[...,1,0]],dim=-1),-1)
    survivors=torch.stack([joints[...,i,j] for i in range(3) for j in range(3) if (i,j) not in [(0,1),(1,0)]],dim=-1)
    log_nc=torch.logsumexp(survivors,-1).clamp(max=0)
    log_ig=joints[...,2,2]-log_nc
    return {'cf':cross.exp().clamp(0,1),'cf_logscore':-log_nc,'log_ignorance':log_ig,'visual_log_mass':v,'text_log_mass':t}


def fuse_evidence_sources(w_plus,w_minus,temperature=1.0,source_dim=0):
    """Exact conjunctive conflict among already-normalized DS sources on same K-frame.
    Each source is defined by its feature-wise positive/negative totals. This is
    inter-source conflict, distinct from each source's internal m+ versus m- CF.
    """
    a=torch.as_tensor(w_plus).to(torch.float64)/temperature
    d=torch.as_tensor(w_minus,device=a.device).to(torch.float64)/temperature
    individual=combine_evidence(a,d)
    merged=combine_evidence(a.sum(source_dim),d.sum(source_dim))
    log_nc=(merged['log_normalizer']-individual['log_normalizer'].sum(source_dim)).clamp(max=0)
    return {'cf':(-torch.expm1(log_nc)).clamp(0,1),'cf_logscore':-log_nc,'log_ignorance':merged['log_ignorance'],'individual':individual,'merged':merged}
