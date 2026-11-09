"""Balanced quotients on orbit bases, without orbit sums or division by |G|."""
from dataclasses import dataclass
import numpy as np
from .css import CSSCode
from .gf2 import binary_matrix
from .algebra import validate_action, validate_equivariance


@dataclass
class OrbitBasis:
    shape: tuple
    index: np.ndarray
    orbits: list
    representatives: list


def orbit_basis(group, right, left, representative="first"):
    if representative not in ("first", "last"):
        raise ValueError("representative must be first or last")
    n, m = right.shape[1], left.shape[1]
    index = np.full((n, m), -1, dtype=int)
    orbits, reps = [], []
    for c in range(n):
        for d in range(m):
            if index[c, d] >= 0:
                continue
            orbit = sorted({(int(right[g, c]), int(left[group.inv[g], d])) for g in range(group.order)})
            k = len(orbits)
            for a, b in orbit:
                if index[a, b] >= 0:
                    raise ValueError("Actions fail to partition the pair basis")
                index[a, b] = k
            orbits.append(orbit)
            reps.append(orbit[0] if representative == "first" else orbit[-1])
    return OrbitBasis((n, m), index, orbits, reps)


@dataclass
class BalancedProduct:
    code: CSSCode
    group: object
    dc: np.ndarray
    dd: np.ndarray
    right: tuple
    left: tuple
    spaces: dict


def balanced_product(dc, dd, group, right, left, representative="first"):
    """Two boundary matrices with right actions on C and left actions on D.

    right=(action_C0, action_C1), left=(action_D0, action_D1).
    Each action is shaped (group order, basis size), entries old -> new.
    Free actions are NOT required for this constructor. Qubit order is 01,10.
    """
    dc, dd = binary_matrix(dc), binary_matrix(dd)
    if len(right) != 2 or len(left) != 2:
        raise ValueError("Supply actions on both chain degrees")
    right = tuple(validate_action(group, a, "right") for a in right)
    left = tuple(validate_action(group, a, "left") for a in left)
    validate_equivariance(dc, *right)
    validate_equivariance(dd, *left)
    spaces = {(p, q): orbit_basis(group, right[p], left[q], representative) for p in (0, 1) for q in (0, 1)}
    n00, n01, n10, n11 = [len(spaces[p, q].orbits) for p, q in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    hx = np.zeros((n00, n01+n10), dtype=np.uint8)
    hzt = np.zeros((n01+n10, n11), dtype=np.uint8)
    for j, (c, d) in enumerate(spaces[0, 1].representatives):
        for v in np.flatnonzero(dd[:, d]):
            hx[spaces[0, 0].index[c, v], j] ^= 1
    for j, (c, d) in enumerate(spaces[1, 0].representatives):
        for v in np.flatnonzero(dc[:, c]):
            hx[spaces[0, 0].index[v, d], n01+j] ^= 1
    for j, (c, d) in enumerate(spaces[1, 1].representatives):
        for v in np.flatnonzero(dc[:, c]):
            hzt[spaces[0, 1].index[v, d], j] ^= 1
        for v in np.flatnonzero(dd[:, d]):
            hzt[n01+spaces[1, 0].index[c, v], j] ^= 1
    code = CSSCode(hx, hzt.T, "BP", {"group_order": group.order, "qubit_order": "01,10"})
    return BalancedProduct(code, group, dc, dd, right, left, spaces)
