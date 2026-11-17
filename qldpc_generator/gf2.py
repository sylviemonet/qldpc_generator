"""Exact binary linear algebra. All bases are returned as rows."""
import numpy as np


def binary_matrix(value, name="matrix"):
    a = np.asarray(value)
    if a.ndim != 2 or not np.all((a == 0) | (a == 1)):
        raise ValueError(f"{name} must be a two-dimensional binary matrix")
    return a.astype(np.uint8, copy=True)


def bit_rows(a):
    return [sum(1 << int(j) for j in np.flatnonzero(row)) for row in binary_matrix(a)]


def from_bits(rows, n):
    return np.array([[(r >> j) & 1 for j in range(n)] for r in rows], dtype=np.uint8).reshape(len(rows), n)


class RowSpace:
    """Incremental echelon basis using Python's arbitrary-width integers."""
    def __init__(self, rows=()):
        self.pivots = {}
        for row in rows:
            self.add(row)

    def reduce(self, row):
        for p in sorted(self.pivots, reverse=True):
            if (row >> p) & 1:
                row ^= self.pivots[p]
        return row

    def add(self, row):
        row = self.reduce(int(row))
        if not row:
            return False
        self.pivots[row.bit_length() - 1] = row
        return True


def rank(a):
    return len(RowSpace(bit_rows(a)).pivots)


def rref(a):
    a = binary_matrix(a)
    pivots, r = [], 0
    for j in range(a.shape[1]):
        indices = np.flatnonzero(a[r:, j])
        if not len(indices):
            continue
        p = r + indices[0]
        a[[r, p]] = a[[p, r]]
        targets = np.flatnonzero(a[:, j])
        targets = targets[targets != r]
        a[targets] ^= a[r]
        pivots.append(j)
        r += 1
        if r == a.shape[0]:
            break
    return a, pivots


def nullspace(a):
    a, pivots = rref(a)
    free = [j for j in range(a.shape[1]) if j not in pivots]
    out = np.zeros((len(free), a.shape[1]), dtype=np.uint8)
    for i, j in enumerate(free):
        out[i, j] = 1
        out[i, pivots] = a[:len(pivots), j]
    return out


def in_rowspace(vector, a):
    a = binary_matrix(a)
    v = binary_matrix(np.asarray(vector).reshape(1, -1), "vector")
    if a.shape[1] != v.shape[1]:
        raise ValueError("Vector and row-space widths differ")
    return RowSpace(bit_rows(a)).reduce(bit_rows(v)[0]) == 0


def quotient_basis(kernel_rows, stabilizer_rows):
    k, s = binary_matrix(kernel_rows), binary_matrix(stabilizer_rows)
    if k.shape[1] != s.shape[1]:
        raise ValueError("Basis widths differ")
    space = RowSpace(bit_rows(s))
    selected = [v for v in bit_rows(k) if space.add(v)]
    return from_bits(selected, k.shape[1])


def inverse(a):
    a = binary_matrix(a)
    n, m = a.shape
    if n != m:
        raise ValueError("Inverse requires a square matrix")
    reduced, pivots = rref(np.hstack([a, np.eye(n, dtype=np.uint8)]))
    if pivots[:n] != list(range(n)):
        raise ValueError("Matrix is singular over GF(2)")
    return reduced[:, n:]
