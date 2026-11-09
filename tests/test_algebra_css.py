from itertools import product
import numpy as np
import pytest
import sympy as sp
from qldpc_generator import CSSCode, subsystem_parameters
from qldpc_generator.gf2 import rank, nullspace, in_rowspace, inverse
from qldpc_generator.algebra import (x, y, polynomial_to_circulant, bivariate_matrix,
    cyclic_coefficients, cyclic_group, expand_group_matrix, formal_adjoint,
    subgroup, symmetric_group_3, regular_matrix, permutation_matrix, FiniteGroup)


def brute_distance(code):
    """Independent exhaustive oracle: explicitly enumerate stabilizer spans."""
    best = code.n+1
    for checks, stabs in ((code.hx, code.hz), (code.hz, code.hx)):
        span = {tuple(np.zeros(code.n, dtype=np.uint8))}
        for row in stabs:
            span |= {tuple(np.array(v, dtype=np.uint8) ^ row) for v in list(span)}
        for bits in product((0,1), repeat=code.n):
            v = np.array(bits, dtype=np.uint8)
            if bits not in span and not np.any(checks @ v % 2):
                best = min(best, sum(bits))
    return best if best <= code.n else None


def test_gf2_not_real_rank():
    a = np.array([[1,1,0],[1,0,1],[0,1,1]], dtype=np.uint8)
    assert np.linalg.matrix_rank(a) == 3
    assert rank(a) == 2
    assert nullspace(a).tolist() == [[1,1,1]]
    assert in_rowspace([1,0,1], a)
    assert not in_rowspace([1,1,1], a)


@pytest.mark.parametrize("shape", [(0,0),(0,4),(3,0),(2,5),(6,3),(6,6)])
def test_binary_linear_algebra(shape):
    rng = np.random.default_rng(24)
    for _ in range(4):
        a = rng.integers(0,2,size=shape,dtype=np.uint8)
        kernel = nullspace(a)
        assert not np.any(a @ kernel.T % 2)
        assert rank(kernel)+rank(a) == a.shape[1]
        if shape[0] == shape[1] and rank(a) == shape[0]:
            assert np.array_equal(a @ inverse(a) % 2, np.eye(shape[0],dtype=int))


def test_shift_reduction_and_adjoint():
    s3 = polynomial_to_circulant(x,3)
    assert s3.tolist() == [[0,0,1],[1,0,0],[0,1,0]]
    assert np.array_equal(polynomial_to_circulant(x**-1,3), s3.T)
    assert not np.any(polynomial_to_circulant(x**6+1+2*x,3))
    assert np.array_equal(polynomial_to_circulant(1+x+x**7,3), polynomial_to_circulant(1,3))
    a, b = 1+x**-2+x**8, x+x**3
    assert np.array_equal(polynomial_to_circulant(a*b,5), polynomial_to_circulant(a,5) @ polynomial_to_circulant(b,5) % 2)
    m = sp.Matrix([[1+x,x**-2,0],[x**7,1,1+x**2]])
    group = cyclic_group(5)
    assert np.array_equal(expand_group_matrix(group,cyclic_coefficients(formal_adjoint(m),5)),
                          expand_group_matrix(group,cyclic_coefficients(m,5)).T)
    h, hi = subgroup(3,6)
    assert all(np.array_equal(a @ b, np.eye(6)) for a,b in zip(h,hi))


@pytest.mark.parametrize("poly,lift", [(x,0),(x,1.5),(sp.sqrt(x),3),(x/2,3),(y,3)])
def test_reject_bad_polynomials(poly,lift):
    with pytest.raises(ValueError):
        polynomial_to_circulant(poly,lift)


def test_nonabelian_regular_action_order_and_paper_matrices():
    g = symmetric_group_3()
    a, b = np.eye(6,dtype=np.uint8)[1], np.eye(6,dtype=np.uint8)[2]
    r = lambda c: regular_matrix(g,c,"right")
    l = lambda c: regular_matrix(g,c,"left")
    e = np.eye(6,dtype=np.uint8)
    assert g.mul(1,2) != g.mul(2,1)
    assert np.array_equal(r(a) @ r(b), r(e[g.mul(2,1)]))
    assert np.array_equal(l(a) @ l(b), l(e[g.mul(1,2)]))
    assert np.array_equal(r(a) @ l(b), l(b) @ r(a))
    # Exact matrix printed in Appendix E, not generated from the same formula.
    P = np.array([[1,1,0,0,0,0],[1,1,0,0,0,0],[0,0,1,0,1,0],
                  [0,0,0,1,0,1],[0,0,1,0,1,0],[0,0,0,1,0,1]])
    Q = np.array([[1,0,1,0,0,0],[0,1,0,0,1,0],[1,0,1,0,0,0],
                  [0,0,0,1,0,1],[0,1,0,0,1,0],[0,0,0,1,0,1]])
    assert np.array_equal(r(e[0]^a),P)
    assert np.array_equal(l(e[0]^b),Q)
    with pytest.raises(ValueError):
        CSSCode(np.hstack([l(a),l(b)]), np.hstack([l(b).T,l(a).T]))


def test_antipode_bivariate():
    f = 1+x**-2*y+x*y**7
    anti = f.subs({x:1/x,y:1/y}, simultaneous=True)
    assert np.array_equal(bivariate_matrix(anti,4,6), bivariate_matrix(f,4,6).T)


def test_invalid_css_and_group():
    for a,b in [([[1]],[[1]]), ([[2]],[[0]]), ([[0]],[[0,0]])]:
        with pytest.raises(ValueError):
            CSSCode(a,b)
    with pytest.raises(ValueError):
        subgroup(4,6)
    with pytest.raises(ValueError):
        FiniteGroup([[0,1],[1,1]])


def test_distance_against_exhaustive_independent_oracle():
    rng = np.random.default_rng(1234)
    for n in range(1,8):
        for _ in range(4):
            hx = rng.integers(0,2,size=(2,n),dtype=np.uint8)
            kernel = nullspace(hx)
            mix = rng.integers(0,2,size=(2,len(kernel)),dtype=np.uint8)
            code = CSSCode(hx, mix @ kernel % 2)
            result = code.distance(n)
            assert result.get("value") == brute_distance(code)
            if code.k:
                lx,lz = code.logical_operators()
                assert lx.shape == lz.shape == (code.k,n)
                assert np.array_equal(lx @ lz.T % 2,np.eye(code.k,dtype=int))
                assert all(code.is_logical(v,"X") for v in lx)
                assert all(code.is_logical(v,"Z") for v in lz)
                for sector, data in result["sectors"].items():
                    if data["status"] == "exact":
                        v=np.zeros(n,dtype=np.uint8); v[data["witness"]]=1
                        assert code.is_logical(v,sector)


def test_distance_excludes_stabilizers_and_budget_is_not_exact():
    from qldpc_generator.constructions import hgp
    h=np.array([[1,1,0],[0,1,1]],dtype=np.uint8)
    code=hgp(h,h)
    assert not code.is_logical(code.hz[0],"Z")
    assert code.distance(2)["lower_bound"] == 3
    assert code.distance(3)["value"] == 3
    partial=code.distance(6,max_subsets=1)
    assert partial["status"] == "bounded" and partial["lower_bound"] == 1


def test_subsystem_bacon_shor_2x2():
    gx=[[1,1,0,0],[0,0,1,1]]
    gz=[[1,0,1,0],[0,1,0,1]]
    p=subsystem_parameters(gx,gz)
    assert (p["n"],p["k"],p["gauge_qubits"]) == (4,1,1)
    assert rank(p["stabilizer_x"]) == rank(p["stabilizer_z"]) == 1
    assert not np.any(p["stabilizer_x"] @ np.array(gz).T % 2)
