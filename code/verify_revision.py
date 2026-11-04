#!/usr/bin/env python3
"""Check explicit channels and protocol distinctions in the 4 November revision.

Python >= 3.10 and NumPy are required. No network access is used. Results are
written beside this file; the original verify_conservation.py is not modified.
Numerical examples supplement, rather than replace, the analytical proofs.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SEED = 20261104
TOL = 5e-11
rng = np.random.default_rng(SEED)
checks: list[dict[str, object]] = []

def check(name: str, lhs, rhs) -> None:
    """Record a finite-array equality with an absolute numerical tolerance."""
    a, b = np.asarray(lhs), np.asarray(rhs)
    if a.shape != b.shape:
        raise ValueError(f'{name}: shape mismatch {a.shape} != {b.shape}')
    error = float(np.max(np.abs(a-b)))
    if not np.isfinite(error) or error > TOL:
        raise AssertionError(f'{name}: residual {error} > {TOL}')
    checks.append({'name':name, 'maximum_absolute_error':error, 'tolerance':TOL})

def density(d: int) -> np.ndarray:
    x = rng.normal(size=(d,d)) + 1j*rng.normal(size=(d,d))
    state = x @ x.conj().T
    return state / np.trace(state)

def unitary(d: int) -> np.ndarray:
    x = rng.normal(size=(d,d)) + 1j*rng.normal(size=(d,d))
    return np.linalg.qr(x)[0]

def channel(kraus: list[np.ndarray], rho: np.ndarray) -> np.ndarray:
    return sum((k @ rho @ k.conj().T for k in kraus), np.zeros_like(rho))

def dual(kraus: list[np.ndarray], observable: np.ndarray) -> np.ndarray:
    return sum((k.conj().T @ observable @ k for k in kraus),
               np.zeros_like(observable))

# The charge-basis measure-and-prepare channel is defined on all density matrices.
e = np.eye(3, dtype=complex)
P = [np.outer(v,v.conj()) for v in e]
K = [P[0], P[2], np.outer(e[0],e[1])/np.sqrt(2),
     np.outer(e[2],e[1])/np.sqrt(2)]
Q = np.diag([-1.,0.,1.]).astype(complex)
check('measure_prepare_trace_preserving', dual(K,np.eye(3)), np.eye(3))
check('measure_prepare_mean_preserving', dual(K,Q), Q)
check('measure_prepare_second_moment', dual(K,Q@Q), np.eye(3))
check('measure_prepare_zero_sector_output', channel(K,P[1]), (P[0]+P[2])/2)
for t in range(20):
    state=density(3)
    expected=(state[0,0]+state[1,1]/2)*P[0]+(state[2,2]+state[1,1]/2)*P[2]
    output=channel(K,state)
    check(f'measure_prepare_all_input_formula_{t}',output,expected)
    check(f'measure_prepare_all_input_mean_{t}',np.trace(Q@output),np.trace(Q@state))

# Sector-diagonal branch instruments can have informative outcome likelihoods.
projectors=[np.diag([1.,1.,0.,0.]),np.diag([0.,0.,1.,1.])]
for t in range(10):
    likelihood=rng.uniform(.1,.9,size=2)
    branches=[]
    for m in range(2):
        k=np.zeros((4,4),complex)
        for q in range(2):
            w=likelihood[q] if m==0 else 1-likelihood[q]
            k[2*q:2*q+2,2*q:2*q+2]=np.sqrt(w)*unitary(2)
        branches.append(k)
    state=density(4)
    prior=np.array([np.trace(p@state).real for p in projectors])
    joint=np.zeros((2,2,2))  # input sector, output sector, outcome
    for q,p in enumerate(projectors):
        for r,s in enumerate(projectors):
            for m,k in enumerate(branches):
                output=k@p@state@p@k.conj().T
                joint[q,r,m]=np.trace(s@output).real
    offdiag=joint[0,1,:]+joint[1,0,:]
    check(f'matched_protocol_normalization_{t}',np.array(joint.sum()),np.array(1.))
    check(f'matched_protocol_no_sector_transfer_{t}',offdiag,np.zeros(2))
    event=joint[:,:,0].sum()
    check(f'matched_protocol_endpoint_equality_{t}',joint[:,:,0].sum(axis=0)/event,
          joint[:,:,0].sum(axis=1)/event)
    check(f'matched_protocol_bayes_{t}',joint[:,:,0].sum(axis=0)/event,
          prior*likelihood/(prior@likelihood))

# One invariant marginal cannot certify sector conservation: a bit flip preserves
# the balanced marginal but exchanges every sharp input sector.
X=np.array([[0.,1.],[1.,0.]])
balanced=np.eye(2)/2
check('flip_balanced_marginal',X@balanced@X.T,balanced)
check('flip_sharp_input',X@np.diag([1.,0.])@X.T,np.diag([0.,1.]))
check('flip_unit_sector_transfer',np.array((X@np.diag([1.,0.])@X.T)[1,1]),np.array(1.))

# Explicit convention: sigma_z |0> = |0>; total J_z |00> = |00>.
sz=np.diag([1.,-1.]); jz=(np.kron(sz,np.eye(2))+np.kron(np.eye(2),sz))/2
v=np.array([1.,0.,0.,0.])
check('spin_zero_zero_total_jz',jz@v,v)

report={'seed':SEED,'tolerance':TOL,'checks_passed':len(checks),'checks_failed':0,
        'maximum_absolute_error':max(float(c['maximum_absolute_error']) for c in checks),
        'checks':checks}
path=Path(__file__).resolve().parent/'revision_results.json'
path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
