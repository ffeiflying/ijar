"""Proposition-specific binary BPAs; exact support/opposition fusion."""
import numpy as np

def binary_totals(positive,negative):
    positive=np.asarray(positive,dtype=np.float64)
    negative=np.asarray(negative,dtype=np.float64)
    base=np.logaddexp(-positive,-negative)
    lognc=base+np.log1p(-np.exp(-positive-negative-base))
    p=(-np.expm1(-positive))*np.exp(-negative-lognc)
    n=(-np.expm1(-negative))*np.exp(-positive-lognc)
    ignorance=np.exp(-positive-negative-lognc)
    internal_cf=(-np.expm1(-positive))*(-np.expm1(-negative))
    assert np.allclose(p+n+ignorance,1,atol=1e-8)
    return {'support':p,'opposition':n,'ignorance':ignorance,'internal_cf':internal_cf,'text_visual_cf':n,'opposite_betp':n+.5*ignorance,'net_negative':negative-positive}

def head_evidence(log_ratios,reliability=None,temperature=1.):
    value=np.asarray(log_ratios,dtype=np.float64)/temperature
    if reliability is None:
        reliability=np.ones_like(value)
    r=np.clip(reliability,0,1)
    committed=r*(-np.expm1(-np.abs(value)))
    effective=-np.log1p(-np.minimum(committed,1-1e-15))
    # Mean canonical weights temper correlated views; duplicating every head is invariant.
    positive=np.where(value>=0,effective,0).mean(0)
    negative=np.where(value<0,effective,0).mean(0)
    result=binary_totals(positive,negative)
    result.update(positive=positive,negative=negative)
    return result

def proposition_pushforward(support,opposition,ignorance):
    """Random-set pushforward: all premises true vs at least one false.

    Product joint source assumption; propositions are different truth variables,
    not falsely exclusive world objects. Correlation is a modeling limitation.
    """
    p=np.asarray(support,dtype=np.float64)
    n=np.asarray(opposition,dtype=np.float64)
    u=np.asarray(ignorance,dtype=np.float64)
    assert np.allclose(p+n+u,1,atol=1e-8)
    log_survive=np.log1p(-np.minimum(n,1-1e-15)).sum()
    conflict=-np.expm1(log_survive)
    normal=np.prod(p)
    unknown=max(0.,1-conflict-normal)
    return {'cf':float(conflict),'cf_logscore':float(-log_survive),'normal_belief':float(normal),'ignorance':float(unknown),'opposite_betp':float(conflict+.5*unknown)}

def verify():
    import itertools
    rng=np.random.default_rng(42)
    maximum=0.
    for _ in range(30):
        logits=rng.normal(size=(4,3))
        masses=np.exp(logits-logits.max(-1,keepdims=True))
        masses/=masses.sum(-1,keepdims=True)
        result=proposition_pushforward(masses[:,0],masses[:,1],masses[:,2])
        direct=np.zeros(3)
        for choices in itertools.product(range(3),repeat=4):
            prob=np.prod([masses[i,j] for i,j in enumerate(choices)])
            cls=1 if 1 in choices else (0 if all(j==0 for j in choices) else 2)
            direct[cls]+=prob
        maximum=max(maximum,abs(direct[0]-result['normal_belief']),abs(direct[1]-result['cf']),abs(direct[2]-result['ignorance']))
    assert maximum<1e-12
    a=head_evidence(np.array([[10.],[-10.]]))
    b=head_evidence(np.array([[0.],[0.]]))
    assert a['net_negative']==b['net_negative']
    assert a['internal_cf']>.98 and b['internal_cf']==0
    return {'random_set_enumeration_cases':30,'max_error':maximum,'same_net_evidence_strong_conflict':float(a['internal_cf'][0]),'same_net_evidence_vacuous_conflict':float(b['internal_cf'][0]),'scope':'mathematical non-equivalence to net contrast; not proof of novelty or semantic validity'}

if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
