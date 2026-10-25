"""Consonant semantic random sets; exact conflict with arbitrary source dependence."""
import numpy as np
from scipy.optimize import linprog

def nested(p,rho=1.):
    p=np.asarray(p,float);p=p/p.sum();order=np.argsort(-p,kind='stable');s=p[order]
    mass=np.arange(1,len(p)+1)*(s-np.r_[s[1:],0.])
    mass*=rho;mass[-1]+=1-rho
    contour=np.empty_like(p);contour[order]=np.cumsum(mass[::-1])[::-1]
    return order,mass,contour

def conflict(a,b):
    order,mass,_=a;_,_,pi=b
    overlap=np.maximum.accumulate(pi[order])
    independent=float(np.sum(mass*(1-overlap)))
    # Nested neighborhoods imply Hall's maximum prefix deficit is exact.
    minimum=float(max(0.,np.max(np.cumsum(mass)-overlap)))
    return minimum,independent

def against_assertion(a,target=0):
    order,mass,pi=a
    rank=int(np.where(order==target)[0][0])
    cf=float(mass[:rank].sum())
    ignorance=float(mass[-1])
    betp=np.zeros(len(mass))
    for i in range(len(mass)):betp[order[:i+1]]+=mass[i]/(i+1)
    return cf,ignorance,float(betp[target])

def verify():
    rng=np.random.default_rng(28);worst=0.;betp_error=0.
    for _ in range(30):
        p=rng.dirichlet(np.ones(5));q=rng.dirichlet(np.ones(5))
        a=nested(p);b=nested(q)
        recovered=np.array([against_assertion(a,i)[2] for i in range(5)])
        betp_error=max(betp_error,float(abs(recovered-p).max()))
        if rng.random()<.5:a=nested(p,.4);b=nested(q,.6)
        oa,ma,_=a;ob,mb,_=b
        cost=np.array([[not set(oa[:i+1])&set(ob[:j+1]) for j in range(5)] for i in range(5)],float)
        eq=np.zeros((10,25))
        for i in range(5):eq[i,i*5:(i+1)*5]=1;eq[5+i,i::5]=1
        opt=linprog(cost.ravel(),A_eq=eq,b_eq=np.r_[ma,mb],bounds=(0,None),method='highs')
        assert opt.success
        lower,ind=conflict(a,b);worst=max(worst,abs(lower-opt.fun))
        assert lower<=ind+1e-12 and abs(conflict(a,a)[0])<1e-12
    assert worst<1e-10 and betp_error<1e-10
    return {'cases':30,'minimum_conflict_vs_transport_lp_max_error':worst,'inverse_pignistic_max_error':betp_error,'same_source_minimum_conflict':0,'interpretation':'minimum empty intersection over all couplings of two random sets; no independence assumed; affinity semantics unverified'}
if __name__=='__main__':print(verify())
