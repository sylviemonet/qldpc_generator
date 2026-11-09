# qldpc_generator

Reproducible finite constructions, code parameters, and basis correspondences accompanying the submitted paper:

> Yuan Yao, Guo Zhang, Yingli Yang, Yuanye Zhu, Xiaosi Xu, Xiao Yuan, and Ying Li, **“An introduction to qLDPC codes from a structural perspective.”** Submitted.

The paper citation and manuscript checksum are recorded in `paper_cases.PAPER_SOURCE`.

The project has three layers: **library functions implement constructions and calculations, automated tests verify them, and notebooks explain and demonstrate them.** Notebooks call the library rather than maintain separate copies of the algorithms.

## Installation and execution

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,notebooks]'
python -m pytest -q
python -m qldpc_generator.verify_paper
python scripts/execute_notebooks.py
```

The last command executes each notebook in a separate Jupyter kernel and saves its outputs. If JupyterLab is already installed, you can also open `notebooks/` interactively. Installing this project does not install JupyterLab itself.

The validation report is written to `reports/paper_validation.json`. Use `--output PATH` to select another location, or `--skip-distance` to run only the algebraic and correspondence checks. The report records software versions, source-code hashes, the paper citation, individual results, and logical-operator witnesses for certified distances.

`requirements-tested.txt` records the dependencies used for the local macOS / Python 3.14 validation. Use `pyproject.toml` for installation on other platforms.

```python
import numpy as np
from qldpc_generator import hgp

H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
code = hgp(H, H)
print(code.parameters())       # N=13, K=1; both check matrices have rank 6
print(code.distance(3))        # Exact D=3, including X/Z logical supports
LX, LZ = code.logical_operators()
assert np.array_equal(LX @ LZ.T % 2, np.eye(code.k, dtype=int))
```

## Implemented constructions and checks

| Topic | Library interface / module | Main checks |
| --- | --- | --- |
| GF(2), CSS, and subsystem codes | `gf2`, `CSSCode`, `subsystem_parameters` | Binary rank, kernels, stabilizer quotients, logical pairing, gauge and stabilizer dimensions |
| Circulants, group algebras, and Tanner graphs | `algebra`, `representations` | Shift direction, negative exponents, binary cancellation, left/right multiplication, labelled graph round trips |
| HGP | `hgp`, `hgp_distance_witness` | Rectangular and rank-deficient inputs, transpose-code dimensions, quotient map in the distance argument |
| LP / LCS | `lifted_product`, `group_lifted_product`, `lcs` | Formal adjoints, shared lift coordinates, nonabelian S3 examples, three large cyclic LP instances |
| FB | `fiber_bundle` | A twist on each nonzero base incidence, separate actions on both fiber chain spaces, chain-automorphism compatibility |
| BP | `balanced_product` | One source representative per orbit, binary induced boundaries, even-order groups, nonfree actions, representative independence |
| Construction correspondences | `lp_to_bp`, `bp_to_lp`, `fb_to_bp`, `bp_to_fb` | Explicit P0/P1/P2; the same P1 for HX and HZ; hypothesis checks for converse constructions |
| Bicycle / GB / BB | `standard_bicycle`, `generalized_bicycle`, `bivariate_bicycle` | Polynomial gcd, independent Groebner quotient dimensions, antipodes, direct-sum decompositions |
| Related variants | `generalized_toric`, `toric`, `coprime_bicycle`, `planar_surface` | Monomial translations, coprime CRT permutations, periodic and open repetition-chain boundaries |
| Paper instances | `paper_cases`, `verify_paper` | Twelve explicitly specified finite instances, with separate distance-claim and computation statuses |

The implementation uses dense binary matrices for auditable finite examples. It is not designed as a generator for extremely large sparse codes or as a decoder-performance benchmark.

## Mathematical conventions and interfaces

- `code.hx` and `code.hz` both have check rows and qubit columns. Construction verifies `HX @ HZ.T % 2 == 0`. Matrix inputs must contain 0/1 entries, and all ranks are computed over GF(2).
- HGP accepts `H1, H2` as classical parity-check matrices. Qubits are ordered CC, then VV.
- The cyclic shift satisfies `S e_j = e_(j+1 mod l)`. Polynomial coefficients are reduced modulo two and exponents modulo the specified period. Integer Laurent exponents are supported.
- LP takes A and B over a shared ring, forms the product, and then expands it into binary blocks. `formal_adjoint` combines matrix transposition with `x -> x^-1`.
- Group-algebra coefficients have shape `(rows, columns, group_order)`. Alternatively, `group_algebra_matrix(group, entries)` accepts a list of group-element indices in each cell; repeated terms cancel. By default, A acts on the right and B on the left. The mitten instance explicitly exchanges the two sides.
- Group-action arrays have shape `(group_order, basis_size)` and map old basis indices to new ones. BP takes `right=(C0,C1)` and `left=(D0,D1)`, allowing different dimensions in different degrees. Action laws and boundary equivariance are checked.
- FB uses `twists[a,b]` for the group label on the incidence `b -> a`. Entries outside the base support must be -1. Omitting twists assigns the identity to every incidence.
- Correspondence permutations map old indices to new indices. `Equivalence.verify()` compares both complete check matrices; equality of N and K alone does not establish the correspondence.

### Migration from the original API

`HGPGenerator.get_parity_matrices()` and `BPGenerator.get_parity_matrices()` retain the historical return order `(HxT, Hz)`. Despite those names, this is **`(HZ.T, HX)` in the paper's convention**. New code should use the functional constructors or `generator.get_css_code().hx/.hz`. `show_parity_matrices()` uses the paper's convention.

The original circulant helper used the opposite shift direction. It now follows `S e_j = e_(j+1)`. Transpose the new matrix if the old array orientation is required, and regenerate saved data that depended on the old indexing. BP outputs can also differ because the incorrect orbit-merging algorithm has been replaced.

The `BPGenerator` compatibility adapter supports square factors whose two chain degrees share the supplied actions. Use `balanced_product()` for the general case. Historical k1/k2 arguments are retained as metadata; actual parameters are recomputed from the matrices.

The original `FBGenerator` was a nonfunctional random-twist prototype. Its replacement has the explicit, reproducible signature `FBGenerator(base, fiber, group, action0, action1, twists=None)` and checks the supplied connection data.

## Interpreting distance results

`CSSCode.distance(max_weight, max_subsets)` performs an exhaustive meet-in-the-middle collision search. Equal check syndromes with unequal stabilizer-quotient signatures identify a nontrivial logical operator. Exhaustion of all smaller supports establishes minimality. Tests cross-check this algorithm against an independent full-space enumeration on small codes.

- `exact`: smaller weights have been excluded and a logical witness has been found for this finite instance. This does not prove a statement for an entire family.
- `bounded`: only a lower bound, or combined lower and upper bounds from the two sectors, has been established.
- `no_logicals`: K=0, so there is no finite nontrivial logical distance.
- `not_searched`: no distance search was performed in this run.
- `reported` / `reported_upper_bound`: a literature value quoted by the paper, stored separately in `distance_claim`; it is not automatically a result certified by this program.
- The general LCS distance expression retains the status `conjectured` in the construction metadata.

The default report certifies exact distances of 3 for HGP13, 4 for LCS52, 2 for bicycle6, 8 for GB48, and 6 for BB72. For BP90, BP108, BB144, and mitten150, it excludes logical operators of weight at most six and reports D>=7. For the three large LP instances, it independently checks the matrices, ranks, and K; the cited distance upper bounds do not yet have local certificates.

Search stops before exceeding the subset budget and reports `checked_through` and the stopping reason. Raising the weight limit can have exponential cost. Failure to find a logical operator within a bounded search does not imply that none exists.

## Example notebooks

1. [HGP and the distance argument](notebooks/01-hgp-and-distance.ipynb)
2. [LP, LCS, and the order of lifting](notebooks/02-lp-lcs.ipynb)
3. [BP orbits and finite instances](notebooks/03-example-bp.ipynb)
4. [FB, LP–BP, and FB–BP correspondences](notebooks/04-fb-and-correspondences.ipynb)
5. [Bicycle codes, BB quotient dimensions, and direct sums](notebooks/05-bicycle-bb.ipynb)
6. [Paper validation summary](notebooks/06-paper-validation.ipynb)

## Scope and missing inputs

The library implements the five core constructions and the listed variants, finite matrix checks, and correspondence hypotheses. It does not independently prove every literature result discussed in the submitted paper.

- The paper quotes GB `[[126,28,8]]` without specifying its two input polynomials.
- The open-boundary `[[88,6,6]]` and tile `[[288,8,12]]` and `[[512,18,19]]` instances require complete boundary-check data. `planar_surface` implements a repetition-chain HGP, not those instances. Explicit HX/HZ matrices can be checked using `CSSCode` and the distance routines once supplied.
- Asymptotic rate/distance theorems, the general LCS distance conjecture, decoding thresholds, single-shot guarantees, and hardware experiments require their respective proofs, simulations, or source data.

See the [repository audit and implementation roadmap](IMPLEMENTATION_PLAN.md). The GitHub Actions workflow runs the tests, report generator, and notebooks; its results are available in the repository's Actions tab and PR checks.
