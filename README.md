# qldpc_generator

Reproducible finite constructions, code parameters, and basis correspondences accompanying the submitted paper:

> Yuan Yao, Guo Zhang, Yingli Yang, Yuanye Zhu, Xiaosi Xu, Xiao Yuan, and Ying Li, **“An introduction to qLDPC codes from a structural perspective.”** Submitted.

The paper citation and manuscript checksum are recorded in `paper_cases.PAPER_SOURCE`.

The project has three layers: **library functions implement constructions and calculations, automated tests verify them, and notebooks explain and demonstrate them.** Notebooks call the library rather than maintain separate copies of the algorithms.

## Start here: a reader's guide

Open [06 — Paper coverage and validation](notebooks/06-paper-validation.ipynb) first. Its tables place each paper distance claim beside the result actually computed here, then identify the correspondence checks and missing inputs. Saved outputs are included; running a notebook regenerates its results.

For the main proof checks, continue with **01 (HGP distance)** and **04 (LP–BP / FB–BP)**. For a construction-by-construction introduction, follow **01 → 02 → 03 → 04 → 05**. The assumed background is binary linear algebra and CSS codes. Each walkthrough states its inputs and hypotheses, gives locally numbered equations, displays intermediate results, and includes failure controls.

| Walkthrough | Paper location | What the reader can inspect |
| --- | --- | --- |
| [01 — HGP and distance](notebooks/01-hgp-and-distance.ipynb) | Sec. V, Eq. (44) and distance argument | Complete [[13,1,3]] matrices; quotient map; a zero-syndrome stabilizer as a failure control; upper bound versus exact distance |
| [02 — LP and LCS](notebooks/02-lp-lcs.ipynb) | Definition 4; Apps. D, G.1–G.2 | All three LP seed matrices; adjoint versus plain transpose; shared lift coordinates; explicit LCS input and [[52,4,4]] certificate |
| [03 — BP orbit bases](notebooks/03-example-bp.ipynb) | Sec. VIII; Apps. G.4–G.5, Algorithm 2 | Source-to-target orbit trace; two incorrect counting rules; BP90/BP108 inputs and independent folded matrices |
| [04 — LP–BP and FB–BP](notebooks/04-fb-and-correspondences.ipynb) | App. F, Propositions 1–2; App. E, Eq. (E9) | Maps on whole orbits; both boundary residuals; a wrong-order map concealed by default representatives; converse hypotheses |
| [05 — Bicycle and BB](notebooks/05-bicycle-bb.ipynb) | Sec. IX, Eqs. (90), (93)–(94) | Polynomial quotient basis; binary ranks; distinction between BB144 and a length-144 direct sum; optional translations and boundary examples |
| [06 — Coverage and results](notebooks/06-paper-validation.ipynb) | Locations linked above; App. G.6 for the optional soundness example | Paper claims versus local certificates/bounds, relation checks, missing data, and reproducibility commands |

Paper numbering refers to the manuscript identified by `PAPER_SOURCE`; notebooks also retain stable LaTeX source labels for later revisions. Their own equation numbers restart in each notebook and are explicitly distinguished from paper numbers.

The default report has **five exact finite distances, four distance lower bounds, and three instances with no local distance search**. All 12 have matrix/rank checks. A passing report does not certify every statement discussed in the review; see [scope and missing inputs](#scope-and-missing-inputs).

## Installation and execution

Use Python 3.10 or newer. Clone the repository (with GitHub access if it is private), then run from its root. If you already have a checkout, start with the virtual-environment command:

```bash
git clone https://github.com/sylviemonet/qldpc_generator.git
cd qldpc_generator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,notebooks]'
python -m pytest -q
python -m qldpc_generator.verify_paper
python scripts/execute_notebooks.py
```

The last command executes each notebook in a fresh Jupyter kernel **using the Python interpreter that invoked the script**, prints its elapsed time, and saves its outputs. It stops on a cell error. This avoids accidentally running an unrelated globally installed `python3` kernel.

For interactive use, open `notebooks/` in JupyterLab or a notebook editor and select the environment where this project was installed. Installing this project's notebook dependencies does not install the JupyterLab application. If needed, install it and register a clearly named kernel from the active environment:

```bash
python -m pip install jupyterlab
python -m ipykernel install --user --name qldpc-generator --display-name "Python (qldpc-generator)"
python -m jupyterlab
```

Choose **Python (qldpc-generator)** and run all cells from the top. Each notebook is independent; no previous notebook's variables are needed. Notebook 06 also offers a `RUN_DISTANCE_SEARCH = False` switch for an algebra-only overview. Default examples are small; raising distance limits can be exponentially expensive.

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

## Scope and missing inputs

The library implements the five core constructions and the listed variants, finite matrix checks, and correspondence hypotheses. It does not independently prove every literature result discussed in the submitted paper.

- The paper quotes GB `[[126,28,8]]` without specifying its two input polynomials.
- The open-boundary `[[88,6,6]]` and tile `[[288,8,12]]` and `[[512,18,19]]` instances require complete boundary-check data. `planar_surface` implements a repetition-chain HGP, not those instances. Explicit HX/HZ matrices can be checked using `CSSCode` and the distance routines once supplied.
- Asymptotic rate/distance theorems, the general LCS distance conjecture, decoding thresholds, single-shot guarantees, and hardware experiments require their respective proofs, simulations, or source data.

See the [repository audit and implementation roadmap](IMPLEMENTATION_PLAN.md). The GitHub Actions workflow runs the tests, report generator, and notebooks; its results are available in the repository's Actions tab and PR checks.
