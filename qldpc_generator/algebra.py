"""Finite groups, permutation actions, and binary Laurent-polynomial lifts."""
from itertools import permutations, product
from math import gcd
import numpy as np
import sympy as sp
from .gf2 import binary_matrix

x, y = sp.symbols("x y")


def positive_integer(n, name="size"):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(n)


class FiniteGroup:
    """Small finite group specified by a verified multiplication table."""
    def __init__(self, table, labels=None):
        t = np.asarray(table)
        if t.ndim != 2 or t.shape[0] != t.shape[1] or len(t) == 0:
            raise ValueError("A nonempty square multiplication table is required")
        n = len(t)
        if not np.issubdtype(t.dtype, np.integer) or np.any((t < 0) | (t >= n)):
            raise ValueError("Group indices must be integers in range")
        self.table, self.order = t.copy(), n
        ids = [e for e in range(n) if np.array_equal(t[e], np.arange(n)) and np.array_equal(t[:, e], np.arange(n))]
        if len(ids) != 1:
            raise ValueError("Group table has no unique identity")
        self.identity = ids[0]
        self.inv = np.empty(n, dtype=int)
        for a in range(n):
            invs = np.flatnonzero((t[a] == self.identity) & (t[:, a] == self.identity))
            if len(invs) != 1:
                raise ValueError("Group table lacks unique inverses")
            self.inv[a] = invs[0]
            for b in range(n):
                if not np.array_equal(t[t[a, b], :], t[a, t[b, :]]):
                    raise ValueError("Group table is not associative")
        self.labels = list(range(n)) if labels is None else list(labels)
        if len(self.labels) != n:
            raise ValueError("One label per group element is required")

    def mul(self, a, b):
        return int(self.table[a, b])

    def regular(self, side="left"):
        if side == "left":
            return self.table.copy()
        if side == "right":
            return self.table.T.copy()
        raise ValueError("side must be left or right")


def cyclic_group(n):
    n = positive_integer(n)
    return FiniteGroup((np.arange(n)[:, None]+np.arange(n)[None, :]) % n)


def symmetric_group_3():
    # Paper order E=(e,s,t,st,ts,sts), s=(12), t=(23).
    elements = [(0, 1, 2), (1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    index = {p: i for i, p in enumerate(elements)}
    return FiniteGroup([[index[tuple(a[b[i]] for i in range(3))] for b in elements] for a in elements], elements)


def direct_product_group(a, b):
    pairs = list(product(range(a.order), range(b.order)))
    return FiniteGroup([[a.mul(i, k)*b.order+b.mul(j, l) for k, l in pairs] for i, j in pairs], pairs)


def permutation_matrix(mapping):
    p = np.asarray(mapping)
    if p.ndim != 1 or not np.issubdtype(p.dtype, np.integer) or sorted(p.tolist()) != list(range(len(p))):
        raise ValueError("Expected a permutation vector (old index -> new index)")
    out = np.zeros((len(p), len(p)), dtype=np.uint8)
    out[p, np.arange(len(p))] = 1
    return out


def permutation_vector(matrix):
    a = binary_matrix(matrix)
    if a.shape[0] != a.shape[1] or not np.all(a.sum(0) == 1) or not np.all(a.sum(1) == 1):
        raise ValueError("Expected a binary permutation matrix")
    return np.argmax(a, axis=0)


def validate_action(group, action, side="left"):
    a = np.asarray(action)
    if a.ndim != 2 or len(a) != group.order:
        raise ValueError("Action must list one permutation per group element")
    for p in a:
        permutation_matrix(p)
    if not np.array_equal(a[group.identity], np.arange(a.shape[1])):
        raise ValueError("Identity must act trivially")
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    for g, h in product(range(group.order), repeat=2):
        composite = a[g][a[h]] if side == "left" else a[h][a[g]]
        if not np.array_equal(composite, a[group.mul(g, h)]):
            raise ValueError(f"Invalid {side} group-action composition law")
    return a.astype(int, copy=True)


def validate_equivariance(boundary, action0, action1):
    d = binary_matrix(boundary)
    if action0.shape[1] != d.shape[0] or action1.shape[1] != d.shape[1] or len(action0) != len(action1):
        raise ValueError("Actions do not match the chain-space dimensions")
    for a0, a1 in zip(action0, action1):
        if not np.array_equal(d[:, a1], d[np.argsort(a0), :]):
            raise ValueError("Boundary is not equivariant under the supplied actions")


def repeated_action(action, copies):
    a = np.asarray(action)
    n = a.shape[1]
    return np.concatenate([a+i*n for i in range(copies)], axis=1) if copies else np.empty((len(a), 0), dtype=int)


def group_algebra_matrix(group, entries):
    """Each cell is an iterable of group indices; duplicate terms cancel."""
    rows = list(entries)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Use a nonempty rectangular array of group-element lists")
    a = np.zeros((len(rows), len(rows[0]), group.order), dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, terms in enumerate(row):
            for g in terms:
                if not isinstance(g, (int, np.integer)) or not 0 <= g < group.order:
                    raise ValueError("Invalid group element")
                a[i, j, g] ^= 1
    return a


def regular_matrix(group, coefficients, side="left"):
    c = np.asarray(coefficients)
    if c.shape != (group.order,) or not np.all((c == 0) | (c == 1)):
        raise ValueError("Expected one binary coefficient per group element")
    out = np.zeros((group.order, group.order), dtype=np.uint8)
    action = group.regular(side)
    for g in np.flatnonzero(c):
        out ^= permutation_matrix(action[g])
    return out


def expand_group_matrix(group, coefficients, side="left"):
    a = np.asarray(coefficients)
    if a.ndim != 3 or a.shape[2] != group.order or not np.all((a == 0) | (a == 1)):
        raise ValueError("Expected a (rows, columns, group order) binary coefficient tensor")
    r, c, q = a.shape
    out = np.zeros((r*q, c*q), dtype=np.uint8)
    for i, j in product(range(r), range(c)):
        out[i*q:(i+1)*q, j*q:(j+1)*q] = regular_matrix(group, a[i, j], side)
    return out


def laurent_terms(poly, variables, periods):
    """Reduce integral Laurent exponents periodically and coefficients mod 2."""
    periods = tuple(positive_integer(n) for n in periods)
    expr = sp.expand(poly.as_expr() if isinstance(poly, sp.Poly) else sp.sympify(poly))
    if expr.free_symbols - set(variables):
        raise ValueError("Unexpected polynomial variables")
    terms = set()
    for term in sp.Add.make_args(expr):
        powers = term.as_powers_dict()
        exps = [powers.get(v, sp.Integer(0)) for v in variables]
        if any(e.is_integer is not True for e in exps):
            raise ValueError("Only integer Laurent exponents are supported")
        coeff = sp.cancel(term / sp.prod(v**e for v, e in zip(variables, exps)))
        if coeff.is_Integer is not True:
            raise ValueError("Polynomial coefficients must be integers")
        if int(coeff) % 2:
            exponent = tuple(int(e) % n for e, n in zip(exps, periods))
            terms.symmetric_difference_update({exponent})
    return sorted(terms)


def polynomial_to_circulant(poly, l):
    l = positive_integer(l)
    out = np.zeros((l, l), dtype=np.uint8)
    for (exponent,) in laurent_terms(poly, (x,), (l,)):
        out[(np.arange(l)+exponent) % l, np.arange(l)] ^= 1
    return out


def cyclic_coefficients(matrix, lift):
    m = sp.Matrix(matrix)
    a = np.zeros((m.rows, m.cols, positive_integer(lift)), dtype=np.uint8)
    for i, j in product(range(m.rows), range(m.cols)):
        for (e,) in laurent_terms(m[i, j], (x,), (lift,)):
            a[i, j, e] = 1
    return a


def formal_adjoint(matrix):
    return sp.Matrix(matrix).T.applyfunc(lambda e: e.subs(x, x**-1))


def bivariate_matrix(poly, L, M):
    L, M = positive_integer(L), positive_integer(M)
    out = np.zeros((L*M, L*M), dtype=np.uint8)
    cols = np.arange(L*M)
    for a, b in laurent_terms(poly, (x, y), (L, M)):
        rows = ((cols//M+a) % L)*M + (cols % M+b) % M
        out[rows, cols] ^= 1
    return out


def subgroup(m, l):
    """Powers x^m,...,1; m must divide l (legacy argument/order convention)."""
    m, l = positive_integer(m, "subgroup step"), positive_integer(l)
    if l % m:
        raise ValueError("The subgroup step must divide the cyclic order")
    matrices = [polynomial_to_circulant(x**(m*i), l) for i in range(1, l//m+1)]
    return matrices, [a.T for a in matrices]
