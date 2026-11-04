#!/usr/bin/env python3
"""Reproduce finite examples for Quantum Measurement and Relational Conservation.

Requires Python >= 3.10 and NumPy. Run from any directory. Results are written
beside this file. The checks test finite implementations of the stated identities;
they are not a substitute for the analytic proofs or experimental evidence.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np

SEED = 20261103
TOL = 5e-11
rng = np.random.default_rng(SEED)
checks: list[dict[str, object]] = []


def check(name: str, lhs, rhs, tol: float = TOL) -> None:
    a, b = np.asarray(lhs), np.asarray(rhs)
    err = float(np.max(np.abs(a - b)))
    checks.append({'name': name, 'maximum_absolute_error': err, 'tolerance': tol})
    if not np.isfinite(err) or err > tol:
        raise AssertionError(f'{name}: error {err:.6g} exceeds {tol:.6g}')


def density(d: int) -> np.ndarray:
    x = rng.normal(size=(d, d)) + 1j*rng.normal(size=(d, d))
    y = x @ x.conj().T
    return y / np.trace(y)


def partial_trace_output(x: np.ndarray, dg: int, db: int) -> np.ndarray:
    """Trace the second factor of a matrix ordered as G tensor B."""
    return np.einsum('aibi->ab', x.reshape(dg, db, dg, db))


def posterior(prior: np.ndarray, likelihood: np.ndarray):
    prob = float(prior @ likelihood)
    if prob <= 0:
        raise ValueError('A conditional distribution requires a positive probability.')
    post = prior * likelihood / prob
    tv = float(np.sum(np.abs(post-prior))/2)
    return prob, post, tv


def rotor_isometry(labels: np.ndarray, outcomes: np.ndarray, c: np.ndarray):
    """No cyclic wraparound: the output basis contains every required charge."""
    pairs = sorted({(int(l-m), int(m)) for l in labels for m in outcomes})
    lookup = {pair: i for i, pair in enumerate(pairs)}
    v = np.zeros((len(pairs), len(labels)), dtype=complex)
    branches = []
    for j, m in enumerate(outcomes):
        k = np.zeros_like(v)
        for i, l in enumerate(labels):
            k[lookup[(int(l-m),int(m))],i] = c[i,j]
        branches.append(k)
        v += k
    return pairs, v, branches


# Sector-preserving rotor preparations, including sector-dependent probabilities.
for t in range(40):
    labels=np.arange(-2,3); outcomes=np.array([-1,0,1])
    c=rng.normal(size=(5,3))+1j*rng.normal(size=(5,3))
    c/=np.linalg.norm(c,axis=1)[:,None]
    pairs,v,branches=rotor_isometry(labels,outcomes,c)
    qout=np.diag([a+b for a,b in pairs]); qin=np.diag(labels)
    rho=density(5); prior=np.real(np.diag(rho))
    check(f'rotor_isometry_{t}',v.conj().T@v,np.eye(5))
    check(f'rotor_intertwining_{t}',qout@v,v@qin)
    avg=np.zeros(5)
    for j,k in enumerate(branches):
        a=k@rho@k.conj().T; pm=np.trace(a).real
        actual=np.array([sum(a[r,r].real for r,(u,w) in enumerate(pairs) if u+w==l) for l in labels])/pm
        expected=prior*np.abs(c[:,j])**2/pm
        check(f'rotor_posterior_{t}_{j}',actual,expected)
        avg+=pm*actual
    check(f'rotor_average_{t}',avg,prior)

# Exact binary example.
p=np.array([.5,.5]); w=np.array([.75,.25]); pm,post,d=posterior(p,w)
check('binary_probability',pm,.5)
check('binary_posterior',post,[.75,.25])
check('binary_tv',d,.25)
check('binary_mean',post@np.array([0.,1.]),.25)
check('binary_covariance_identity',(post-p)@np.array([0.,1.]),
      ((p*np.array([0.,1.]))@w-(p@np.array([0.,1.]))*(p@w))/(p@w))

# The sharp worst-prior bound: endpoint priors attain it.
bounds=[]
for a,b in [(0.75,0.25),(0.9,0.1),(0.6,0.4),(1.0,0.01),(0.2,0.05)]:
    r=math.sqrt(b)/(math.sqrt(a)+math.sqrt(b)); bound=(math.sqrt(a)-math.sqrt(b))/(math.sqrt(a)+math.sqrt(b))
    event,_,tv=posterior(np.array([r,1-r]),np.array([a,b]))
    check(f'sharp_bound_{a}_{b}',tv,bound)
    check(f'sharp_probability_{a}_{b}',event,math.sqrt(a*b))
    bounds.append({'a':a,'b':b,'prior_at_a':r,'event_probability':event,'maximum_tv':bound})
for t in range(100):
    w=rng.uniform(.01,.99,7); p=rng.dirichlet(np.ones(7)); prob,post,tv=posterior(p,w)
    a,b=float(max(w)),float(min(w)); bound=(math.sqrt(a)-math.sqrt(b))/(math.sqrt(a)+math.sqrt(b))
    check(f'bound_inequality_{t}',max(0.,tv-bound),0.)
    check(f'tv_identity_{t}',tv,float(p@np.abs(w-prob))/(2*prob))
    check(f'variance_bound_{t}',max(0.,tv-math.sqrt(float(p@((w-prob)**2)))/(2*prob)),0.)

# Finite references and normalized states, with exact overlaps (zero extension).
finite=[]
for N in [1,3,11,51,201]:
    J=(N-1)//2; labels=np.arange(-J,J+1); outcomes=np.array([0,1])
    c=np.ones((N,2))/math.sqrt(2); pairs,v,ks=rotor_isometry(labels,outcomes,c)
    phi=np.ones(N)/math.sqrt(N); omega=v@phi
    rs=np.zeros((2,2),complex)
    for i,(a,m) in enumerate(pairs):
        for j,(b,n) in enumerate(pairs):
            if a==b:rs[m,n]+=omega[i]*omega[j].conj()
    gamma=max(0.,1-1/N); target=np.ones((2,2))/2
    dist=np.sum(np.abs(np.linalg.eigvalsh(rs-target)))/2
    purity=np.trace(rs@rs).real; variance=float(np.mean(labels**2))
    check(f'finite_rho_{N}',rs,[[.5,gamma/2],[gamma/2,.5]])
    check(f'finite_distance_{N}',dist,1/(2*N))
    check(f'finite_purity_{N}',purity,(1+gamma**2)/2)
    check(f'finite_variance_{N}',variance,(N*N-1)/12)
    for m,k in enumerate(ks):
        branch=k@phi*math.sqrt(2); meanp=sum(abs(branch[i])**2*a for i,(a,n) in enumerate(pairs))
        check(f'finite_preparer_shift_{N}_{m}',meanp,-m)
    finite.append({'N':N,'gamma':gamma,'system_trace_distance':float(dist),'purity':float(purity),'preparer_variance_hbar2':variance})

# Scalar effects: no conditional change in an arbitrary ancillary state.
for t in range(20):
    dg,dp=3,4; state=density(dg*dp)
    x=rng.normal(size=(dp,dp))+1j*rng.normal(size=(dp,dp)); u,_=np.linalg.qr(x)
    k=math.sqrt(.37)*u; op=np.kron(np.eye(dg),k)
    out=op@state@op.conj().T
    check(f'ancilla_scalar_effect_{t}',partial_trace_output(out,dg,dp)/np.trace(out),partial_trace_output(state,dg,dp))
# Sector-dependent effect steers a classically correlated ancilla conditionally.
rho=np.diag([.5,0.,0.,.5]); op=np.kron(np.eye(2),np.diag(np.sqrt([.75,.25])))
out=op@rho@op.conj().T
check('ancilla_conditional_counterexample',partial_trace_output(out,2,2)/np.trace(out),np.diag([.75,.25]))

# Phase-sensitive output states with identical likelihoods.
labels=np.array([0,1]); outcomes=np.array([0,1]); c0=np.ones((2,2))/math.sqrt(2); c1=c0.copy();c1[1,1]*=-1
states=[]
for c in [c0,c1]:
    pairs,v,ks=rotor_isometry(labels,outcomes,c);state=v@(np.ones(2)/math.sqrt(2));rs=np.zeros((2,2),complex)
    for i,(a,m) in enumerate(pairs):
        for j,(b,n) in enumerate(pairs):
            if a==b:rs[m,n]+=state[i]*state[j].conj()
    states.append(rs)
check('phase_positive_coherence',states[0][0,1],.25)
check('phase_negative_coherence',states[1][0,1],-.25)

# Two-spin singlet/triplet example preserves each total component nonselectively.
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]],complex);sz=np.diag([1.,-1.]);eye=np.eye(2)
singlet=np.array([0,1,-1,0],complex)/math.sqrt(2);Ps=np.outer(singlet,singlet.conj());Pt=np.eye(4)-Ps
for name,s in [('x',sx),('y',sy),('z',sz)]:
    j=(np.kron(s,eye)+np.kron(eye,s))/2
    check(f'spin_commutator_{name}',Ps@j-j@Ps,np.zeros((4,4)))
    check(f'spin_nonselective_{name}',Ps@j@Ps+Pt@j@Pt,j)

# Additive kinetic energy change of two rotors in one basis component.
for l,m,Ip,Is in [(2,1,3.,5.),(-3,2,1.,2.),(0,1,1.,1.)]:
    delta=(l-m)**2/(2*Ip)+m*m/(2*Is)-l*l/(2*Ip)
    formula=-l*m/Ip+m*m*(1/Ip+1/Is)/2
    check(f'kinetic_energy_{l}_{m}',delta,formula)

root=Path(__file__).resolve().parent
report={'seed':SEED,'tolerance':TOL,'checks_passed':len(checks),'checks_failed':0,
        'maximum_absolute_error':max(x['maximum_absolute_error'] for x in checks),
        'finite_reference_table':finite,'sharp_bound_examples':bounds,'checks':checks}
(root/'computed_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
with (root/'finite_reference_table.csv').open('w',newline='',encoding='utf8') as f:
    writer=csv.DictWriter(f,fieldnames=list(finite[0]));writer.writeheader();writer.writerows(finite)
print(json.dumps({k:report[k] for k in ['seed','checks_passed','checks_failed','maximum_absolute_error']},indent=2))
print('Finite reference table:')
for row in finite:print(row)
