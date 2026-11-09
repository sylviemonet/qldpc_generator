"""Transcribed finite inputs from clean.tex (2026-11-07); expectations are separate.

Distance literature values remain reported until an exhaustive calculation or
a checked logical witness establishes the corresponding statement.
"""
from dataclasses import dataclass
import numpy as np
import sympy as sp
from .algebra import (x, y, cyclic_group, symmetric_group_3, direct_product_group,
                      group_algebra_matrix, formal_adjoint, polynomial_to_circulant)
from .constructions import (hgp, lifted_product, lcs, standard_bicycle, generalized_bicycle,
                            bivariate_bicycle, group_lifted_product)
from .balanced import balanced_product
from .css import CSSCode
from .equivalences import Equivalence

PAPER_SOURCE = {"filename": "clean.tex", "version": "2026-11-07",
                "sha256": "09a2db27ae7e3d15404a0c1fa4e979d2c12857f5eae5d159c385e44f83188e93"}

LP_EXPONENTS = {
    16: [[0,0,0,0,0], [0,2,4,7,11], [0,3,10,14,15]],
    21: [[0,0,0,0,0], [0,4,5,7,17], [0,14,18,12,11]],
    30: [[0,0,0,0,0], [0,2,14,24,25], [0,16,11,14,13]],
}


def cyclic_lp_case(lift):
    m = sp.Matrix([[x**e for e in row] for row in LP_EXPONENTS[lift]])
    return lifted_product(m, formal_adjoint(m), lift)


def cyclic_bp_case(q, representative="first"):
    if q not in (5, 6):
        raise ValueError("The paper supplies q=5 and q=6")
    n = 3*q
    p1 = 1+x+x**5
    p2 = 1+x**2+x**7 if q == 5 else 1+x+x**2
    action = np.array([[(i+3*g) % n for i in range(n)] for g in range(q)])
    return balanced_product(polynomial_to_circulant(p1, n), polynomial_to_circulant(p2, n),
                            cyclic_group(q), (action, action), (action, action), representative)


def s3_case():
    group = symmetric_group_3()
    return group_lifted_product(group, group_algebra_matrix(group, [[[0, 1]]]),
                                group_algebra_matrix(group, [[[0, 2]]]))


def mitten_case():
    group = direct_product_group(cyclic_group(5), symmetric_group_3())
    U = group_algebra_matrix(group, [[[0, 6+4, 12+5], [0, 6, 5]]])
    VT = group_algebra_matrix(group, [[[6+3, 12+4, 24+3]], [[0, 6, 24+5]]])
    return group_lifted_product(group, U, VT, a_side="left")


def bb_family(t):
    return bivariate_bicycle(x**(3*t)+y+y**2, y**3+x**t+x**(2*t), 6*t, 6)


def bb_direct_sum_equivalence(t):
    code, block = bb_family(t), bb_family(1)
    target = CSSCode(np.kron(np.eye(t, dtype=np.uint8), block.hx),
                     np.kron(np.eye(t, dtype=np.uint8), block.hz), "direct sum of BB72")
    checks = np.array([((i % t)*36+(i//t)*6+j) for i in range(6*t) for j in range(6)])
    qubits = np.array([((i % t)*72+s*36+(i//t)*6+j) for s in (0,1) for i in range(6*t) for j in range(6)])
    return Equivalence(code, target, checks, qubits, checks, "i=r+tu: t independent [[72,12,6]] blocks")


def bb144_case():
    return bivariate_bicycle(y**2*(x+x**2+y**3), x**2*(y+y**2+x**3), 12, 6)


@dataclass(frozen=True)
class PaperCase:
    identifier: str
    build: object
    n: int
    k: int
    rank_x: int
    rank_z: int
    distance_value: int | None
    distance_kind: str
    paper_label: str
    source: str


def paper_cases():
    h = np.array([[1,1,0], [0,1,1]], dtype=np.uint8)
    return [
        PaperCase("hgp13", lambda: hgp(h,h), 13,1,6,6,3,"exact_finite_claim","sec:HGP","tillich2014"),
        PaperCase("lp544", lambda: cyclic_lp_case(16), 544,80,232,232,12,"reported_upper_bound","app:lp-examples","xu2024constant"),
        PaperCase("lp714", lambda: cyclic_lp_case(21), 714,100,307,307,16,"reported_upper_bound","app:lp-examples","xu2024constant"),
        PaperCase("lp1020", lambda: cyclic_lp_case(30), 1020,136,442,442,20,"reported_upper_bound","app:lp-examples","xu2024constant"),
        PaperCase("lcs52", lambda: lcs(2,4), 52,4,24,24,4,"exact_finite_claim","app:lcs-details","old2024lift"),
        PaperCase("bp90", lambda: cyclic_bp_case(5).code, 90,8,41,41,10,"reported","app:bp-examples","tiew2024lowoverheadentanglinggatesgeneralised"),
        PaperCase("bp108", lambda: cyclic_bp_case(6).code, 108,8,50,50,8,"reported","app:bp-examples","tiew2024lowoverheadentanglinggatesgeneralised"),
        PaperCase("bicycle6", lambda: standard_bicycle(1+x,3), 6,2,2,2,2,"exact_finite_claim","sec:BB","mackay2004sparse"),
        PaperCase("gb48", lambda: generalized_bicycle(1+x**2+x**8+x**15,1+x**2+x**12+x**17,24),48,6,21,21,8,"reported","sec:BB","panteleev2021degenerate"),
        PaperCase("bb72", lambda: bb_family(1),72,12,30,30,6,"reported","eq:bb-logical-qubit-quotient","bravyi2024high"),
        PaperCase("bb144", bb144_case,144,12,66,66,12,"reported","eq:fg","bravyi2024high"),
        PaperCase("mitten150", mitten_case,150,30,60,60,10,"reported","eq:mitten-regular-checks","bhardwaj2026highrate"),
    ]
