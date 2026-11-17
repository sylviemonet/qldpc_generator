"""Explicit basis bijections for the paper's LP--BP and FB--BP propositions."""
from dataclasses import dataclass
from math import gcd
import numpy as np
from .algebra import (expand_group_matrix, repeated_action, permutation_matrix,
                      permutation_vector, bivariate_matrix, x, y)
from .balanced import balanced_product
from .constructions import group_lifted_product, fiber_bundle


@dataclass
class Equivalence:
    source: object
    target: object
    p0: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    description: str = "basis permutation"

    def verify(self):
        """p maps source index -> target index; the SAME p1 is used twice."""
        for p, size in zip((self.p0, self.p1, self.p2), (len(self.source.hx), self.source.n, len(self.source.hz))):
            permutation_matrix(p)
            if len(p) != size:
                return False
        hx = self.source.hx[np.argsort(self.p0)][:, np.argsort(self.p1)]
        hz = self.source.hz[np.argsort(self.p2)][:, np.argsort(self.p1)]
        return np.array_equal(hx, self.target.hx) and np.array_equal(hz, self.target.hz)


def _degree_maps(bp, maps, order="01,10"):
    """maps[p,q] lists each BP orbit's target tensor-coordinate index."""
    a, b = ((0, 1), (1, 0)) if order == "01,10" else ((1, 0), (0, 1))
    # Source BP is always 01,10; target offsets depend on its block order.
    offsets = {a: 0, b: len(maps[a])}
    p1 = np.r_[np.asarray(maps[0, 1])+offsets[0, 1], np.asarray(maps[1, 0])+offsets[1, 0]]
    return np.asarray(maps[0, 0], dtype=int), p1.astype(int), np.asarray(maps[1, 1], dtype=int)


def lp_to_bp(group, A, B, representative="first"):
    """Construct BP independently and certify Phi([(alpha,x),(beta,y)])=(alpha,beta,yx)."""
    ra, na, _ = np.shape(A)
    rb, nb, _ = np.shape(B)
    # These symmetry actions are not the boundary multiplication actions.
    right = group.regular("left")[group.inv]
    left = group.regular("right")[group.inv]
    bp = balanced_product(expand_group_matrix(group, A, "right"), expand_group_matrix(group, B, "left"), group,
                          (repeated_action(right, ra), repeated_action(right, na)),
                          (repeated_action(left, rb), repeated_action(left, nb)), representative)
    target = group_lifted_product(group, A, B)
    maps = {}
    q = group.order
    for (p, r), space in bp.spaces.items():
        nd = (rb, nb)[r]
        maps[p, r] = [(c//q*nd+d//q)*q+group.mul(d % q, c % q) for c, d in space.representatives]
    eq = Equivalence(bp.code, target, *_degree_maps(bp, maps, "10,01"), "LP--BP: k=yx")
    if not eq.verify():
        raise AssertionError("LP--BP chain-map identity failed")
    return bp, eq


def _free_coordinates(group, action, inverse_coordinate=True):
    """Choose orbit reps and identify each free orbit with G."""
    n = action.shape[1]
    coordinates = np.full((n, 2), -1, dtype=int)
    representatives = []
    for v in range(n):
        if coordinates[v, 0] >= 0:
            continue
        orbit = action[:, v]
        if len(set(orbit.tolist())) != group.order:
            raise ValueError("This correspondence requires free actions on every chosen basis")
        i = len(representatives)
        representatives.append(v)
        for g in range(group.order):
            index = group.inv[g] if inverse_coordinate else g
            coordinates[action[index, v]] = (i, g)
    return representatives, coordinates


def bp_to_lp(bp):
    """Converse LP--BP; reject nonfree actions, extract the group-ring entries."""
    group = bp.group
    cr = [_free_coordinates(group, a) for a in bp.right]
    dr = [_free_coordinates(group, a) for a in bp.left]
    def coefficients(boundary, coords):
        a = np.zeros((len(coords[0][0]), len(coords[1][0]), group.order), dtype=np.uint8)
        for j, rep in enumerate(coords[1][0]):
            for v in np.flatnonzero(boundary[:, rep]):
                i, g = coords[0][1][v]
                a[i, j, g] ^= 1
        return a
    A, B = coefficients(bp.dc, cr), coefficients(bp.dd, dr)
    target = group_lifted_product(group, A, B)
    maps = {}
    for (p, q), space in bp.spaces.items():
        arr = []
        for c, d in space.representatives:
            i, a = cr[p][1][c]
            j, b = dr[q][1][d]
            arr.append((i*len(dr[q][0])+j)*group.order+group.mul(b, a))
        maps[p, q] = arr
    eq = Equivalence(bp.code, target, *_degree_maps(bp, maps, "10,01"), "free BP -> group LP")
    if not eq.verify():
        raise AssertionError("BP--LP converse failed")
    return A, B, eq


def fb_to_bp(base, fiber, group, action0, action1, twists=None, representative="first"):
    """Cover boundary (b,g) -> sum_a (a,phi(b,a)g), T([(b,g),f])=(b,gf)."""
    target = fiber_bundle(base, fiber, group, action0, action1, twists)
    base, fiber = np.asarray(base), np.asarray(fiber)
    if twists is None:
        twists = np.where(base, group.identity, -1)
    twists = np.asarray(twists)
    r, n = base.shape
    q = group.order
    cover = np.zeros((r*q, n*q), dtype=np.uint8)
    for a, b in zip(*np.nonzero(base)):
        for g in range(q):
            cover[a*q+group.mul(twists[a, b], g), b*q+g] ^= 1
    right = group.regular("right")
    bp = balanced_product(cover, fiber, group, (repeated_action(right, r), repeated_action(right, n)),
                          (action0, action1), representative)
    maps = {}
    for (p, s), space in bp.spaces.items():
        act = bp.left[s]
        maps[p, s] = [(c//q)*act.shape[1]+int(act[c % q, f]) for c, f in space.representatives]
    eq = Equivalence(bp.code, target, *_degree_maps(bp, maps), "FB--BP: T([(b,g),f])=(b,gf)")
    if not eq.verify():
        raise AssertionError("FB--BP chain-map identity failed")
    return bp, eq


def bp_to_fb(bp):
    """Sufficient converse: C free, and at most one group label per base incidence.

    Checks this hypothesis for the deterministic chosen orbit representatives;
    does not claim arbitrary BP admits an FB description.
    """
    group = bp.group
    cr = [_free_coordinates(group, a, inverse_coordinate=False) for a in bp.right]
    base = np.zeros((len(cr[0][0]), len(cr[1][0])), dtype=np.uint8)
    twists = np.full(base.shape, -1, dtype=int)
    for j, rep in enumerate(cr[1][0]):
        for v in np.flatnonzero(bp.dc[:, rep]):
            i, g = cr[0][1][v]
            if base[i, j]:
                raise ValueError("Boundary violates the single-group-label covering hypothesis")
            base[i, j], twists[i, j] = 1, g
    target = fiber_bundle(base, bp.dd, group, *bp.left, twists)
    maps = {}
    for (p, q), space in bp.spaces.items():
        maps[p, q] = []
        for c, d in space.representatives:
            b, g = cr[p][1][c]
            maps[p, q].append(b*bp.left[q].shape[1]+int(bp.left[q][g, d]))
    eq = Equivalence(bp.code, target, *_degree_maps(bp, maps), "covering BP -> FB")
    if not eq.verify():
        raise AssertionError("BP--FB converse failed")
    return base, twists, eq


def coprime_gb_bb_equivalence(source, target, L, M):
    if gcd(L, M) != 1 or source.n != 2*L*M:
        raise ValueError("Coprime periods and matching lengths required")
    p = np.array([(k % L)*M+k % M for k in range(L*M)])
    return Equivalence(source, target, p, np.r_[p, p+L*M], p, "CRT: k -> (k mod L,k mod M)")


def monomial_bb_equivalence(source, target, L, M, f_shift, g_shift):
    """Target f=U*source f, target g=V*source g, with monomial shifts U,V."""
    U = bivariate_matrix(x**f_shift[0]*y**f_shift[1], L, M)
    V = bivariate_matrix(x**g_shift[0]*y**g_shift[1], L, M)
    n = L*M
    return Equivalence(source, target, permutation_vector(U),
                       np.r_[np.arange(n), permutation_vector(V.T @ U)+n],
                       permutation_vector(V.T), "BB monomial-template translations")
