from fractions import Fraction
import numpy as np
import pytest
from qldpc_generator.paper_cases import (paper_cases, cyclic_bp_case, bb_family,
                                        bb144_case, bb_direct_sum_equivalence)
from qldpc_generator.constructions import (generalized_bicycle, bivariate_bicycle, bb_quotient_dimension,
    gb_dimension, coprime_bicycle, generalized_toric, toric, planar_surface)
from qldpc_generator.algebra import x,y,polynomial_to_circulant
from qldpc_generator.equivalences import coprime_gb_bb_equivalence,monomial_bb_equivalence
from qldpc_generator.representations import tanner_edges,tanner_matrix,repetition_checks,classical_soundness


@pytest.mark.parametrize("case",paper_cases(),ids=lambda c:c.identifier)
def test_published_shapes_and_binary_ranks(case):
    code=case.build(); p=code.parameters()
    assert (p["n"],p["k"],p["rank_x"],p["rank_z"])==(case.n,case.k,case.rank_x,case.rank_z)
    assert np.array_equal(tanner_matrix(code.hx.shape,tanner_edges(code.hx)),code.hx)


@pytest.mark.parametrize("identifier,weight",[("hgp13",3),("lcs52",4),("bicycle6",2),("gb48",8),("bb72",6)])
def test_certified_finite_distances(identifier,weight):
    code=next(c for c in paper_cases() if c.identifier==identifier).build()
    result=code.distance(weight)
    assert result["status"]=="exact" and result["value"]==weight
    for sector,data in result["sectors"].items():
        if data["status"]=="exact":
            v=np.zeros(code.n,dtype=np.uint8); v[data["witness"]]=1
            assert code.is_logical(v,sector) and int(v.sum())==data["value"]


@pytest.mark.parametrize("q",[5,6])
def test_bp_independent_folded_matrices_and_representatives(q):
    n=3*q
    pc=lambda p:polynomial_to_circulant(p,n)
    p2=1+x**2+x**7 if q==5 else 1+x+x**2
    a=np.block([[pc(1),pc(x**6),pc(x**3)],[pc(1),pc(1),pc(x**6)],[pc(x**3),pc(1),pc(1)]])
    b=np.kron(np.eye(3,dtype=np.uint8),pc(p2))
    code=cyclic_bp_case(q).code
    assert np.array_equal(code.hx,np.hstack([b,a]))
    assert np.array_equal(code.hz,np.hstack([a.T,b.T]))
    last=cyclic_bp_case(q,"last").code
    assert np.array_equal(code.hx,last.hx) and np.array_equal(code.hz,last.hz)


@pytest.mark.parametrize("lift",[1,2,4,5,6,9])
def test_gb_gcd_even_and_odd_periods(lift):
    for a,b in [(0,0),(1+x,1+x**2),(1+x+x**3,1+x**4)]:
        code=generalized_bicycle(a,b,lift)
        assert code.k==gb_dimension(a,b,lift)


@pytest.mark.parametrize("L,M,f,g",[(2,2,1+x,1+y),(3,4,1+x+y,1+x*y),
    (6,6,x**3+y+y**2,y**3+x+x**2),(12,6,y**2*(x+x**2+y**3),x**2*(y+y**2+x**3))])
def test_bb_independent_groebner_quotient(L,M,f,g):
    code=bivariate_bicycle(f,g,L,M)
    assert code.k==2*bb_quotient_dimension(f,g,L,M)


@pytest.mark.parametrize("t",[1,2,3,4])
def test_bb_family_direct_sum_with_same_qubit_map(t):
    eq=bb_direct_sum_equivalence(t)
    assert eq.verify()
    assert eq.source.n==72*t and eq.source.k==12*t
    assert np.all(eq.source.hx.sum(1)==6) and np.all(eq.source.hz.sum(1)==6)


def test_coprime_and_monomial_equivalences():
    source=generalized_bicycle(1+x+x**5,1+x**2,15)
    target=coprime_bicycle(1+x+x**5,1+x**2,3,5)
    assert coprime_gb_bb_equivalence(source,target,3,5).verify()
    with pytest.raises(ValueError,match="Coprime"):
        coprime_bicycle(1+x,1+x,4,6)
    target=generalized_toric(-1,-3,3,-1,12,6)
    assert monomial_bb_equivalence(bb144_case(),target,12,6,(-1,-2),(-2,-1)).verify()


def test_periodic_and_open_boundaries_are_distinct_constructions():
    periodic=toric(3,3)
    open_code=planar_surface(3,3)
    assert (periodic.n,periodic.k,periodic.distance(3)["value"])==(18,2,3)
    assert (open_code.n,open_code.k,open_code.distance(3)["value"])==(13,1,3)


@pytest.mark.parametrize("n",[4,6,8])
def test_path_star_soundness_same_code_different_presentation(n):
    path,star=repetition_checks(n),repetition_checks(n,"star")
    assert classical_soundness(path)==Fraction(2,n-1)
    assert classical_soundness(star)==Fraction(n,n-1)
    assert int(path.sum(0).max())==2 and int(star.sum(0).max())==n-1
