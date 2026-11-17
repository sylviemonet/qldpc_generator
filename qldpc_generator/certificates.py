"""Small, explicit certificates for the HGP distance argument."""
import numpy as np
from .gf2 import binary_matrix, rank, nullspace, in_rowspace
from .constructions import hgp


def classical_distance(checks, max_dimension=20):
    """Exact kernel enumeration. Zero-dimensional code has distance infinity."""
    checks=binary_matrix(checks)
    basis=nullspace(checks)
    if len(basis)>max_dimension:
        raise ValueError("Kernel exceeds the configured exhaustive-enumeration dimension")
    if not len(basis):
        return {"status":"exact", "value":float("inf"), "witness":None}
    v=np.zeros(checks.shape[1],dtype=np.uint8)
    best=checks.shape[1]+1
    witness=None
    # Gray-code enumeration changes exactly one basis row at each step.
    for i in range(1,1 << len(basis)):
        j=(i & -i).bit_length()-1
        v ^= basis[j]
        weight=int(v.sum())
        if weight<best:
            best,witness=weight,v.copy()
    return {"status":"exact", "value":best, "witness":witness}


def hgp_distance_witness(H):
    """Symmetric full-row-rank HGP: certify z=(0,c tensor e_j) is logical.

    Returns a finite upper-bound certificate. The general HGP lower bound
    remains a mathematical theorem; use CSSCode.distance for small instances.
    """
    H=binary_matrix(H)
    r,n=H.shape
    if rank(H)!=r or n==r:
        raise ValueError("This argument requires full row rank and a nonzero classical code")
    classical=classical_distance(H)
    e=np.eye(n,dtype=np.uint8)
    j=next(j for j in range(n) if not in_rowspace(e[j],H))
    z=np.r_[np.zeros(r*r,dtype=np.uint8),np.kron(classical["witness"],e[j])]
    q=nullspace(H)
    quotient=np.kron(np.eye(n,dtype=np.uint8),q)
    code=hgp(H,H)
    assert code.is_logical(z,"Z")
    assert not np.any(quotient @ code.hz[:,r*r:].T % 2)
    assert np.any(quotient @ z[r*r:] % 2)
    return {"code":code,"logical":z,"coordinate":j,"quotient_map":q,
            "upper_bound":int(z.sum()),"classical_distance":classical["value"]}
