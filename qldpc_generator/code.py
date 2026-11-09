"""Compatibility adapters.

Legacy HGP/BP get_parity_matrices() returns (paper HZ.T, paper HX), despite
the historical names HxT and Hz. New code should use get_css_code() or the
functional constructors and access .hx/.hz in the paper convention.
"""
import numpy as np
from .algebra import FiniteGroup, permutation_vector
from .constructions import hgp, fiber_bundle
from .balanced import balanced_product
from .gf2 import binary_matrix


class QLDPCGenerator:
    def get_css_code(self):
        raise NotImplementedError("Choose a concrete construction")

    def show_code_parameters(self):
        return self.get_css_code().parameters()

    def show_parity_matrices(self):
        code = self.get_css_code()
        return code.hx, code.hz


class HGPGenerator(QLDPCGenerator):
    def __init__(self, n1, k1, H1, n2, k2, H2T):
        self.H1, self.H2T = binary_matrix(H1), binary_matrix(H2T)
        if self.H1.shape[1] != n1 or self.H2T.shape[0] != n2:
            raise ValueError("n1/n2 do not match H1/H2T")
        self.n1, self.k1, self.n2, self.k2 = n1, k1, n2, k2
        self.r1, self.r2 = self.H1.shape[0], self.H2T.shape[1]
        self.HxT, self.HZ = None, None
        self._code = None

    def get_css_code(self):
        if self._code is None:
            self._code = hgp(self.H1, self.H2T.T)
        return self._code

    def _calculate_HGP_parameters(self):
        code = self.get_css_code()
        return code.n, code.k

    def get_parity_matrices(self):
        if self.HxT is None:
            code = self.get_css_code()
            self.HxT, self.Hz = code.hz.T, code.hx
            self.HZ = self.Hz
        return self.HxT, self.Hz


class BPGenerator(HGPGenerator):
    """Legacy square-input cyclic/permutation API.

    Use balanced_product() for rectangular inputs, distinct actions in the two
    degrees, nonfree actions, or explicit nonabelian left/right conventions.
    k1/k2 are retained as historical metadata; parameters are recomputed.
    """
    def __init__(self, n1, k1, H1, n2, k2, H2T, h1, h1inv=None, h2=None):
        super().__init__(n1, k1, H1, n2, k2, H2T)
        if self.r1 != n1 or self.r2 != n2:
            raise ValueError("Legacy BP needs square factors; use balanced_product for separate degree actions")
        self.h1 = [binary_matrix(a) for a in h1]
        self.h2 = self.h1 if h2 is None else [binary_matrix(a) for a in h2]
        self.h1inv = [a.T for a in self.h1]
        self.h2inv = [a.T for a in self.h2]
        if h1inv is not None and (len(h1inv) != len(self.h1) or any(not np.array_equal(a, b) for a, b in zip(h1inv, self.h1inv))):
            raise ValueError("Supplied inverse actions do not match permutation transposes")
        right = np.array([permutation_vector(a) for a in self.h1])
        left = np.array([permutation_vector(a) for a in self.h2])
        lookup = {tuple(p): i for i, p in enumerate(right)}
        if len(lookup) != len(right):
            raise ValueError("Legacy group list contains duplicates")
        try:
            table = [[lookup[tuple(h[g])] for h in right] for g in right]
        except KeyError as e:
            raise ValueError("Legacy group list is not closed") from e
        self._group = FiniteGroup(table)
        self._right, self._left = right, left

    def get_css_code(self):
        if self._code is None:
            self._balanced = balanced_product(self.H1, self.H2T, self._group,
                                               (self._right, self._right), (self._left, self._left))
            self._code = self._balanced.code
        return self._code


class FBGenerator(QLDPCGenerator):
    """Completed FB API; replaces the previously nonfunctional random-twist stub."""
    def __init__(self, base, fiber, group, action0, action1, twists=None):
        self._code = fiber_bundle(base, fiber, group, action0, action1, twists)

    def get_css_code(self):
        return self._code

    def get_parity_matrices(self):
        return self._code.hx, self._code.hz
