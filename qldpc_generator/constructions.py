"""Paper conventions: both HX and HZ have check rows and qubit columns."""
from itertools import product
from math import gcd
import numpy as np
import sympy as sp
from .css import CSSCode
from .gf2 import binary_matrix, rank
from .algebra import (x, y, positive_integer, cyclic_group, cyclic_coefficients,
                      expand_group_matrix, formal_adjoint, polynomial_to_circulant,
                      bivariate_matrix, laurent_terms, validate_action, validate_equivariance,
                      permutation_matrix)


def hgp(H1, H2):
    """HGP of r1 x n1 and r2 x n2 checks; qubits ordered CC then VV."""
    a, b = binary_matrix(H1), binary_matrix(H2)
    r1, n1 = a.shape
    r2, n2 = b.shape
    hx = np.hstack([np.kron(np.eye(r1, dtype=np.uint8), b.T), np.kron(a, np.eye(n2, dtype=np.uint8))])
    hz = np.hstack([np.kron(a.T, np.eye(r2, dtype=np.uint8)), np.kron(np.eye(n1, dtype=np.uint8), b)])
    k1, k2 = n1-rank(a), n2-rank(b)
    kt1, kt2 = r1-rank(a), r2-rank(b)
    return CSSCode(hx, hz, "HGP", {"expected_k": k1*k2+kt1*kt2, "qubit_order": "CC,VV"})


def _kron_identity(a, size, left=False):
    r, c, q = a.shape
    out = np.zeros((r*size, c*size, q), dtype=np.uint8)
    for i, j, k in product(range(r), range(c), range(size)):
        u, v = (k*r+i, k*c+j) if left else (i*size+k, j*size+k)
        out[u, v] = a[i, j]
    return out


def group_lifted_product(group, A, B, a_side="right"):
    """Group LP, default A right/B left; a_side='left' swaps both actions."""
    if a_side not in ("left", "right"):
        raise ValueError("a_side must be left or right")
    b_side = "left" if a_side == "right" else "right"
    A, B = np.asarray(A), np.asarray(B)
    # Validate tensors before deriving dimensions.
    expand_group_matrix(group, A, a_side)
    expand_group_matrix(group, B, b_side)
    ra, na, _ = A.shape
    rb, nb, _ = B.shape
    hx = np.hstack([expand_group_matrix(group, _kron_identity(A, rb), a_side),
                    expand_group_matrix(group, _kron_identity(B, ra, left=True), b_side)])
    hz = np.hstack([expand_group_matrix(group, _kron_identity(B, na, left=True), b_side).T,
                    expand_group_matrix(group, _kron_identity(A, nb), a_side).T])
    return CSSCode(hx, hz, "group LP", {"group_order": group.order, "qubit_order": "10,01", "a_side": a_side})


def lifted_product(A, B, lift):
    """Cyclic LP: form the ring product before expanding the shared lift."""
    code = group_lifted_product(cyclic_group(lift), cyclic_coefficients(A, lift), cyclic_coefficients(B, lift))
    code.name = "cyclic LP"
    return code


def lcs(ell, L):
    ell, L = positive_integer(ell), positive_integer(L)
    a = sp.zeros(ell, ell+1)
    for i in range(ell):
        a[i, i], a[i, i+1] = 1, 1+x
    code = lifted_product(formal_adjoint(a), a, L)
    cut = ell*ell*L
    p = np.r_[np.arange(cut, code.n), np.arange(cut)]
    return CSSCode(code.hx[:, p], code.hz[:, p], "LCS",
                   {"ell": ell, "L": L, "expected_k": L,
                    "distance_claim": {"status": "conjectured", "value": min(L, 2*ell+1)}})


def fiber_bundle(base, fiber, group, action0, action1, twists=None):
    """Twist each nonzero base incidence; allow different fiber actions in degrees 0 and 1.

    twists[a,b] is the group index for incidence b -> a; off-support entries
    must be -1. If omitted, every incidence carries the identity.
    """
    base, fiber = binary_matrix(base), binary_matrix(fiber)
    a0, a1 = validate_action(group, action0), validate_action(group, action1)
    validate_equivariance(fiber, a0, a1)
    r, n = base.shape
    f0, f1 = fiber.shape
    if twists is None:
        twists = np.where(base, group.identity, -1)
    twists = np.asarray(twists)
    if twists.shape != base.shape or not np.issubdtype(twists.dtype, np.integer):
        raise ValueError("Twists must be an integer array with the base shape")
    if np.any(twists[base == 0] != -1) or np.any((twists[base == 1] < 0) | (twists[base == 1] >= group.order)):
        raise ValueError("Every base incidence needs one group label; off-support use -1")
    d0, d1 = np.zeros((r*f0, n*f0), dtype=np.uint8), np.zeros((r*f1, n*f1), dtype=np.uint8)
    for a, b in zip(*np.nonzero(base)):
        g = twists[a, b]
        d0[a*f0:(a+1)*f0, b*f0:(b+1)*f0] = permutation_matrix(a0[g])
        d1[a*f1:(a+1)*f1, b*f1:(b+1)*f1] = permutation_matrix(a1[g])
    hx = np.hstack([np.kron(np.eye(r, dtype=np.uint8), fiber), d0])
    hz = np.vstack([d1, np.kron(np.eye(n, dtype=np.uint8), fiber)]).T
    return CSSCode(hx, hz, "FB", {"qubit_order": "01,10"})


def bicycle(A, B):
    A, B = binary_matrix(A), binary_matrix(B)
    if A.shape[0] != A.shape[1] or B.shape != A.shape:
        raise ValueError("Bicycle blocks must be square and the same size")
    if np.any((A @ B + B @ A) % 2):
        raise ValueError("Bicycle blocks must commute")
    return CSSCode(np.hstack([A, B]), np.hstack([B.T, A.T]), "bicycle")


def standard_bicycle(a, lift):
    A = polynomial_to_circulant(a, lift)
    return bicycle(A, A.T)


def generalized_bicycle(a, b, lift):
    code = bicycle(polynomial_to_circulant(a, lift), polynomial_to_circulant(b, lift))
    code.name = "GB"
    code.metadata["gcd_k"] = gb_dimension(a, b, lift)
    return code


def gb_dimension(a, b, lift):
    lift = positive_integer(lift)
    polys = [sp.Poly(sum(x**e[0] for e in laurent_terms(p, (x,), (lift,))), x, modulus=2) for p in (a, b)]
    return 2*int(sp.gcd(sp.gcd(*polys), sp.Poly(x**lift+1, x, modulus=2)).degree())


def bivariate_bicycle(f, g, L, M):
    code = bicycle(bivariate_matrix(f, L, M), bivariate_matrix(g, L, M))
    code.name = "BB"
    code.metadata.update({"L": int(L), "M": int(M), "boundary": "periodic"})
    return code


def bb_quotient_dimension(f, g, L, M):
    """Independent Groebner/standard-monomial calculation, not CSS ranks."""
    L, M = positive_integer(L), positive_integer(M)
    polys = [sum(x**a*y**b for a, b in laurent_terms(p, (x, y), (L, M))) for p in (f, g)]
    basis = sp.groebner([*polys, x**L+1, y**M+1], x, y, modulus=2)
    leading = [p.monoms()[0] for p in basis.polys]
    return sum(not any(i >= a and j >= b for a, b in leading) for i in range(L) for j in range(M))


def generalized_toric(a, b, c, d, L, M):
    return bivariate_bicycle(1+x+x**a*y**b, 1+y+x**c*y**d, L, M)


def toric(L, M):
    return bivariate_bicycle(1+x, 1+y, L, M)


def coprime_bicycle(a, b, L, M):
    L, M = positive_integer(L), positive_integer(M)
    if gcd(L, M) != 1:
        raise ValueError("Coprime periods are needed for a single cycle of length LM")
    return bivariate_bicycle(sp.sympify(a).subs(x, x*y), sp.sympify(b).subs(x, x*y), L, M)


def planar_surface(rows, columns):
    """Open repetition-chain HGP; not a truncation of arbitrary BB checks."""
    rows, columns = positive_integer(rows), positive_integer(columns)
    def repetition(n):
        a = np.zeros((n-1, n), dtype=np.uint8)
        for i in range(n-1):
            a[i, i:i+2] = 1
        return a
    return hgp(repetition(rows), repetition(columns))
