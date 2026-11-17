"""Labelled Tanner incidence and finite check-presentation experiments."""
from fractions import Fraction
from itertools import product
import numpy as np
from .gf2 import binary_matrix, nullspace, bit_rows


def tanner_edges(matrix):
    return [tuple(map(int, edge)) for edge in zip(*np.nonzero(binary_matrix(matrix)))]


def tanner_matrix(shape, edges):
    out = np.zeros(shape, dtype=np.uint8)
    seen = set()
    for i, j in edges:
        if not 0 <= i < shape[0] or not 0 <= j < shape[1] or (i,j) in seen:
            raise ValueError("Tanner edges must be distinct, in-range check/qubit pairs")
        out[i, j] = 1
        seen.add((i,j))
    return out


def repetition_checks(n, presentation="path"):
    if not isinstance(n, int) or n < 2 or presentation not in ("path", "star"):
        raise ValueError("Use n>=2 and path or star")
    a = np.zeros((n-1, n), dtype=np.uint8)
    for i in range(n-1):
        a[i, i if presentation == "path" else 0] = 1
        a[i, i+1] = 1
    return a


def classical_soundness(matrix, max_bits=12):
    """Exhaust normalized distance-to-kernel soundness on small classical inputs."""
    a = binary_matrix(matrix)
    m, n = a.shape
    if n > max_bits or not m or not n:
        raise ValueError("This exhaustive helper needs 1<=n<=max_bits and nonempty checks")
    kernel = bit_rows(nullspace(a))
    codewords = [0]
    for v in kernel:
        codewords += [w ^ v for w in codewords]
    rows = bit_rows(a)
    best = None
    for v in range(1 << n):
        d = min((v ^ w).bit_count() for w in codewords)
        if d:
            syndrome = sum((v & row).bit_count() % 2 for row in rows)
            ratio = Fraction(syndrome*n, m*d)
            best = ratio if best is None else min(best, ratio)
    return best
