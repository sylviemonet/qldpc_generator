"""CSS validation and finite distance certificates; no floating-point ranks."""
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
import numpy as np
from .gf2 import binary_matrix, rank, nullspace, quotient_basis, inverse, bit_rows, in_rowspace


@dataclass
class CSSCode:
    hx: np.ndarray
    hz: np.ndarray
    name: str = "CSS"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.hx = binary_matrix(self.hx, "HX")
        self.hz = binary_matrix(self.hz, "HZ")
        if self.hx.shape[1] != self.hz.shape[1]:
            raise ValueError("HX and HZ must have the same number of columns")
        if np.any((self.hx @ self.hz.T) % 2):
            raise ValueError("CSS checks do not commute over GF(2)")

    @property
    def n(self):
        return self.hx.shape[1]

    @property
    def k(self):
        return self.n - rank(self.hx) - rank(self.hz)

    def parameters(self):
        rx, rz = rank(self.hx), rank(self.hz)
        return {"n": self.n, "k": self.n-rx-rz, "rank_x": rx, "rank_z": rz,
                "checks_x": len(self.hx), "checks_z": len(self.hz),
                "max_row_weight": int(max(self.hx.sum(1).max(initial=0), self.hz.sum(1).max(initial=0))),
                "max_column_weight": int(max(self.hx.sum(0).max(initial=0), self.hz.sum(0).max(initial=0)))}

    def logical_operators(self):
        lx = quotient_basis(nullspace(self.hz), self.hx)
        lz = quotient_basis(nullspace(self.hx), self.hz)
        lx = (inverse((lx @ lz.T) % 2) @ lx) % 2
        return lx, lz

    def is_logical(self, vector, sector="Z"):
        checks, stabilizers = self._sector(sector)
        v = binary_matrix(np.asarray(vector).reshape(1, -1), "logical candidate")[0]
        if len(v) != self.n:
            raise ValueError("Wrong logical-vector length")
        return not np.any(checks @ v % 2) and not in_rowspace(v, stabilizers)

    def _sector(self, sector):
        if sector == "Z":
            return self.hx, self.hz
        if sector == "X":
            return self.hz, self.hx
        raise ValueError("sector must be X or Z")

    def distance(self, max_weight=None, max_subsets=None, *, method="mitm",
                 time_limit=None, threads=None, solver_log=False):
        """Meet-in-the-middle exhaustive search, with explicit stopping bounds.

        A collision of equal check syndromes with unequal stabilizer-quotient
        signatures is a nontrivial logical. To certify odd weight 2t-1, compare
        subsets of size t with sizes <=t-1; to certify 2t, also compare size t
        pairs. Repeated signatures can be discarded without losing existence.
        ``max_subsets`` bounds enumerated subsets per sector, not elapsed time.
        The default method is ``mitm`` (weight 6, one million subsets).
        Use ``method='gurobi'`` for unrestricted-weight MILP minimization with
        a time limit per sector (default 60 seconds) and optional thread count.
        Gurobi is optional; requesting it never silently falls back to MITM.
        """
        if method == "gurobi":
            if max_weight is not None or max_subsets is not None:
                raise ValueError("Gurobi minimizes over all weights; use time_limit, not MITM budgets")
            from .gurobi_distance import gurobi_distance
            return gurobi_distance(self, time_limit=60 if time_limit is None else time_limit,
                                   threads=1 if threads is None else threads, solver_log=solver_log)
        if method != "mitm":
            raise ValueError("method must be mitm or gurobi")
        if time_limit is not None or threads is not None or solver_log:
            raise ValueError("time_limit, threads, and solver_log apply only to method='gurobi'")
        max_weight = 6 if max_weight is None else max_weight
        max_subsets = 1_000_000 if max_subsets is None else max_subsets
        if not isinstance(max_weight, int) or max_weight < 0 or max_subsets < 1:
            raise ValueError("Use a nonnegative weight and positive subset budget")
        if self.k == 0:
            return {"status": "no_logicals", "value": None, "sectors": {}}
        sectors = {s: self._distance_sector(s, min(max_weight, self.n), max_subsets) for s in ("X", "Z")}
        exact = [v["value"] for v in sectors.values() if v["status"] == "exact"]
        lower = min(v["lower_bound"] for v in sectors.values())
        if exact and lower >= min(exact):
            return {"status": "exact", "value": min(exact), "sectors": sectors}
        return {"status": "bounded", "lower_bound": lower,
                "upper_bound": min(exact) if exact else None, "sectors": sectors}

    def _distance_sector(self, sector, bound, budget):
        checks, stabs = self._sector(sector)
        syndromes = bit_rows(checks.T)
        signatures = bit_rows(nullspace(stabs).T)
        # mask is the support, signature detects membership in row(stabs).
        known = {0: (0, 0)}
        used, certified = 1, 0

        def found(sig, mask, record):
            old_sig, old_mask = record
            if old_sig == sig:
                return None
            support = mask ^ old_mask
            witness = [i for i in range(self.n) if (support >> i) & 1]
            return {"status": "exact", "value": len(witness), "lower_bound": len(witness),
                    "witness": witness, "subsets": used}

        for t in range(1, (bound + 1)//2 + 1):
            if used + comb(self.n, t) > budget:
                break
            layer = []
            for indices in combinations(range(self.n), t):
                syn = sig = mask = 0
                for i in indices:
                    syn ^= syndromes[i]
                    sig ^= signatures[i]
                    mask |= 1 << i
                used += 1
                if syn in known:
                    result = found(sig, mask, known[syn])
                    if result:
                        return result
                layer.append((syn, sig, mask))
            certified = 2*t-1
            if certified == bound:
                break
            for syn, sig, mask in layer:
                if syn in known:
                    result = found(sig, mask, known[syn])
                    if result:
                        return result
                else:
                    known[syn] = (sig, mask)
            certified = 2*t
        return {"status": "lower_bound", "lower_bound": certified+1,
                "checked_through": certified, "subsets": used,
                "stop_reason": "weight_limit" if certified >= bound else "subset_budget"}


def subsystem_parameters(gx, gz):
    """CSS subsystem dimensions and central stabilizers from gauge rows."""
    gx, gz = binary_matrix(gx), binary_matrix(gz)
    if gx.shape[1] != gz.shape[1]:
        raise ValueError("Gauge widths differ")
    cross = gx @ gz.T % 2
    gauge_qubits = rank(cross)
    sx = nullspace(cross.T) @ gx % 2
    sz = nullspace(cross) @ gz % 2
    return {"n": gx.shape[1], "k": gx.shape[1]-rank(gx)-rank(gz)+gauge_qubits,
            "gauge_qubits": gauge_qubits, "stabilizer_x": sx, "stabilizer_z": sz}
