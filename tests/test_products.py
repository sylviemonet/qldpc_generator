from itertools import product
import numpy as np
import pytest
from qldpc_generator import hgp, lifted_product, lcs, fiber_bundle, balanced_product
from qldpc_generator.gf2 import rank, nullspace, in_rowspace
from qldpc_generator.algebra import (x, cyclic_group, symmetric_group_3, group_algebra_matrix,
    repeated_action, polynomial_to_circulant, permutation_matrix, permutation_vector)
from qldpc_generator.equivalences import lp_to_bp, bp_to_lp, fb_to_bp, bp_to_fb, Equivalence


@pytest.mark.parametrize("shapes", [((2,3),(3,4)),((3,2),(1,5)),((0,2),(2,3)),((3,3),(2,2))])
def test_hgp_dimensions_rank_and_dependent_rows(shapes):
    rng=np.random.default_rng(77)
    a,b=[rng.integers(0,2,size=s,dtype=np.uint8) for s in shapes]
    if len(a)>1: a[-1]=a[0]
    r1,n1=a.shape; r2,n2=b.shape
    code=hgp(a,b)
    assert code.n==n1*n2+r1*r2
    assert code.hx.shape==(r1*n2,code.n)
    assert code.hz.shape==(n1*r2,code.n)
    assert code.k==(n1-rank(a))*(n2-rank(b))+(r1-rank(a))*(r2-rank(b))


def test_hgp_distance_proof_quotient_witness():
    h=np.array([[1,1,0],[0,1,1]],dtype=np.uint8)
    c=np.ones(3,dtype=np.uint8)
    code=hgp(h,h)
    q=nullspace(h)  # Kernel(q)=row(H), so q represents the quotient map.
    assert not np.any(q @ h.T % 2)
    j=next(j for j in range(3) if not in_rowspace(np.eye(3,dtype=np.uint8)[j],h))
    z=np.r_[np.zeros(4,dtype=np.uint8),np.kron(c,np.eye(3,dtype=np.uint8)[j])]
    assert code.is_logical(z,"Z") and int(z.sum())==3
    assert np.any(np.kron(np.eye(3,dtype=np.uint8),q) @ z[4:] % 2)
    assert not np.any(np.kron(np.eye(3,dtype=np.uint8),q) @ code.hz[:,4:].T % 2)
    assert code.distance(3)["value"]==3


def test_lift_one_hgp_permutation_and_shared_lift_size():
    a=np.array([[1,1,0],[0,1,1]],dtype=np.uint8)
    b=np.array([[1,1,1,0]],dtype=np.uint8)
    hp=hgp(a,b)
    lp=lifted_product(a.tolist(),b.T.tolist(),1)
    cut=a.shape[1]*b.shape[1]
    p1=np.r_[np.arange(cut,lp.n),np.arange(cut)]
    assert np.array_equal(hp.hx,lp.hx[:,p1])
    assert np.array_equal(hp.hz,lp.hz[:,p1])
    small=lifted_product([[1+x]],[[1+x**2]],3)
    big=hgp(polynomial_to_circulant(1+x,3),polynomial_to_circulant(1+x**2,3).T)
    assert small.n==6 and big.n==18


@pytest.mark.parametrize("ell,L", [(1,1),(1,2),(2,3),(2,4),(3,4),(3,5)])
def test_lcs_rank_formula_only(ell,L):
    code=lcs(ell,L)
    assert code.n==((ell+1)**2+ell**2)*L and code.k==L
    assert code.metadata["distance_claim"]["status"]=="conjectured"


@pytest.mark.parametrize("order", [2,4])
def test_even_orbit_order_and_mod2_cancellation(order):
    g=cyclic_group(order)
    right=left=g.regular()
    # D has a trivial action. All coefficients of a C column enter one target orbit.
    trivial=np.zeros((order,1),dtype=int)
    identity=np.eye(order,dtype=np.uint8)
    bp=balanced_product(identity,[[0]],g,(right,right),(trivial,trivial))
    assert bp.code.hx.tolist()==[[0,1]]  # Summing all source reps would incorrectly give zero.
    all_ones=np.ones((order,order),dtype=np.uint8)
    cancelled=balanced_product(all_ones,[[0]],g,(right,right),(trivial,trivial))
    assert not np.any(cancelled.code.hx)  # OR-ing repeated incidences would incorrectly give one.
    assert not np.any(cancelled.code.hz)


@pytest.mark.parametrize("group", [cyclic_group(2),cyclic_group(4),symmetric_group_3()])
def test_lp_bp_both_directions_with_representative_change(group):
    a=group_algebra_matrix(group,[[[0,1],[1]],[[0],[]]])
    b=group_algebra_matrix(group,[[[0],[0,group.order-1]]])
    first,eq=lp_to_bp(group,a,b)
    last,last_eq=lp_to_bp(group,a,b,representative="last")
    assert eq.verify() and last_eq.verify()
    assert np.array_equal(first.code.hx,last.code.hx)
    assert np.array_equal(first.code.hz,last.code.hz)
    aa,bb,reverse=bp_to_lp(first)
    assert np.array_equal(aa,a) and np.array_equal(bb,b) and reverse.verify()
    broken=Equivalence(eq.source,eq.target,eq.p0,eq.p1[::-1],eq.p2)
    assert not broken.verify()


def test_fb_bp_nonfree_fiber_and_converse():
    g=symmetric_group_3()
    # F1 is the natural, nonfree 3-point S3 action; F0 is a fixed point.
    a0=np.zeros((6,1),dtype=int); a1=np.array(g.labels)
    base=np.array([[1,1,0],[0,1,1]],dtype=np.uint8)
    twists=np.array([[1,3,-1],[-1,2,4]])
    bp,eq=fb_to_bp(base,[[1,1,1]],g,a0,a1,twists)
    last,eq_last=fb_to_bp(base,[[1,1,1]],g,a0,a1,twists,representative="last")
    assert eq.verify() and eq_last.verify()
    b,t,reverse=bp_to_fb(bp)
    assert reverse.verify() and np.array_equal(b,base) and np.array_equal(t,twists)
    with pytest.raises(ValueError,match="free"):
        bp_to_lp(bp)
    # Rectangular fiber requires separate degree-zero/one action dimensions.
    bad=np.tile(np.arange(3),(6,1))
    with pytest.raises(ValueError,match="equivariant"):
        fiber_bundle([[1]],np.eye(3,dtype=np.uint8),g,bad,a1)


def test_fb_free_fiber_lp_and_untwisted_hgp():
    g=symmetric_group_3(); act=g.regular("left")
    f=np.eye(6,dtype=np.uint8)
    bp,eq=fb_to_bp([[1,1]],f,g,act,act,[[1,2]])
    assert eq.verify() and bp_to_lp(bp)[2].verify()
    untwisted=fiber_bundle([[1,1]],f,g,act,act)
    hp=hgp([[1,1]],f.T)
    assert np.array_equal(untwisted.hx,hp.hx)
    assert np.array_equal(untwisted.hz,hp.hz)


def test_reject_noncover_bp_and_invalid_actions():
    g=cyclic_group(2); act=g.regular()
    bp=balanced_product(np.ones((2,2),dtype=np.uint8),np.eye(2,dtype=np.uint8),g,(act,act),(act,act))
    with pytest.raises(ValueError,match="single-group-label"):
        bp_to_fb(bp)
    with pytest.raises(ValueError,match="equivariant"):
        balanced_product([[1,0],[0,0]],np.eye(2,dtype=np.uint8),g,(act,act),(act,act))
    with pytest.raises(ValueError,match="incidence"):
        fiber_bundle([[1,0]],np.eye(2,dtype=np.uint8),g,act,act,[[0,0]])
    s3=symmetric_group_3()
    # Treating a right action as a left action must fail on a nonabelian group.
    with pytest.raises(ValueError,match="composition"):
        fiber_bundle([[1]],np.eye(6,dtype=np.uint8),s3,s3.regular("right"),s3.regular("right"))


def test_lp_converse_with_independent_basis_relabeling():
    g=symmetric_group_3()
    a=group_algebra_matrix(g,[[[0,1],[]]])
    b=group_algebra_matrix(g,[[[0,2]]])
    bp,_=lp_to_bp(g,a,b)
    rng=np.random.default_rng(3)
    pc=[rng.permutation(v.shape[1]) for v in bp.right]
    pd=[rng.permutation(v.shape[1]) for v in bp.left]
    rebase=lambda acts,p: p[acts[:,np.argsort(p)]]
    dc=bp.dc[np.argsort(pc[0])][:,np.argsort(pc[1])]
    dd=bp.dd[np.argsort(pd[0])][:,np.argsort(pd[1])]
    shuffled=balanced_product(dc,dd,g,tuple(rebase(v,p) for v,p in zip(bp.right,pc)),
                              tuple(rebase(v,p) for v,p in zip(bp.left,pd)))
    assert bp_to_lp(shuffled)[2].verify()
