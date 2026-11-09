# Paper validation: repository audit and implementation roadmap

This implementation accompanies the submitted paper:

> Yuan Yao, Guo Zhang, Yingli Yang, Yuanye Zhu, Xiaosi Xu, Xiao Yuan, and Ying Li, **“An introduction to qLDPC codes from a structural perspective.”** Submitted.

## Original repository

The audit baseline is commit `84e8f67` on main. The original repository had seven tracked files: an empty README, LICENSE, three library Python files, and two test files. It had no dependency declaration, installation configuration, CI workflow, or current notebooks. Commit `bf36c04` had removed `qldpc_generator/Untitled.ipynb`; there was no `example-bp.ipynb` in the baseline tree.

The baseline test run had **three tests: two passed and one failed**.

| Area | Finding | Impact and resolution |
| --- | --- | --- |
| HGP caching | A NumPy array was used directly in `if array` | Repeated calls failed; replaced with explicit cache-state checks |
| HGP parameters | Undefined rank and H2 references; transpose-code dimensions confused with ranks | Use binary ranks and transpose-kernel dimensions; cover rank-deficient inputs |
| Circulants | No periodic reduction of high powers, no binary coefficient reduction, opposite shift direction | Add Laurent/periodic reduction and a documented shift convention with migration guidance |
| BP boundaries | Entries were set to one when encountered, rather than XORed from a single source representative | Incorrect even-order and repeated-orbit contributions; replace the induced-boundary construction |
| BP actions | No separate actions by chain degree, action-law checks, or equivariance checks | Distinguish C0/C1/D0/D1 and left/right actions explicitly |
| FB | Missing validation methods and incomplete random masks and array dimensions | Replace with explicit incidence twists satisfying chain-map compatibility |
| Test coverage | Only basic shape and commutation checks; no distance, quotient, or correspondence tests | Add independent enumeration, Groebner calculations, explicit basis permutations, and negative controls |

## Library functions and notebooks

Construction formulas, binary linear algebra, group actions, and certificates belong in reusable library functions. Tests, the command-line report, and notebooks call the same functions. Notebooks explain the input data, construction steps, and meaning of the results; automated tests check that later changes preserve those results. A saved notebook output alone cannot cover invalid hypotheses or enforce consistent conventions across examples.

## Implementation milestones and acceptance criteria

| Phase | Implementation | Acceptance criteria | Status |
| --- | --- | --- | --- |
| 1. Foundations and migration | GF(2), CSS/subsystem parameters, logical bases, cyclic/group algebra, legacy HGP/BP return order | Binary versus real rank, empty kernels, dimensions, invalid inputs, API regression checks | Implemented |
| 2. Core constructions | HGP, cyclic/nonabelian LP, LCS, FB, BP, bicycle/GB/BB | Commutation, rectangular and rank-deficient inputs, distinct fiber-degree actions, even-order orbit cancellation | Implemented |
| 3. Correspondences | LP↔BP, FB↔BP, GB↔coprime BB, BB monomial translations and direct sums | Explicit P0/P1/P2, shared P1 for both matrices, C2/C4/S3 examples, rejection of invalid hypotheses | Implemented |
| 4. Finite instances and certificates | Twelve specified instances, five exact distances, bounded searches for other instances | Compare N/K/ranks with independently recorded expectations; verify witnesses; preserve literature-only statuses | Implemented for the available inputs |
| 5. Examples and reproducibility | Six notebooks, JSON report, dependency declarations, CI | Execute notebooks in fresh kernels; rerun tests and reports with documented commands | Implemented; CI results are attached to each run |

Core construction functions are in `constructions.py`, `balanced.py`, and `equivalences.py`. `paper_cases.py` stores the paper inputs and expected values separately from their calculations. Reports refer to section and equation labels rather than page numbers.

## Coverage of the main mathematical arguments

- **HGP distance:** verify the kernel condition, nonstabilizer property, and weight of `z=(0,c⊗e_j)`. Check that the quotient map kills the stabilizer block but not z. Exhaustively verify D=3 for the small example.
- **LP–BP:** construct the factor boundaries and free actions, implement `k=yx` on every orbit, and compare both complete check matrices. S3, independent basis relabelling, and alternative representatives test multiplication order and basis choices.
- **FB–BP:** build the covering complex from the twists and use `[(b,g),f] -> b⊗gf` for the basis map, including nonfree fiber actions. The converse checks the free covering factor and single-group-label incidence hypothesis for the chosen representatives.
- **BP orbit sums:** C2/C4 cases distinguish a single source representative from a sum over the source orbit. Another control distinguishes XOR from OR. The q=5/6 instances are compared entry by entry with the folded matrices printed in the paper.
- **BB logical dimension:** compare GF(2) ranks with an independent Groebner standard-monomial count. Verify the complete matrix permutations for t=1..4 direct sums and certify distance six for the BB72 base block.

## Remaining implementation roadmap

1. **Obtain missing inputs.** Acquire the GB126 polynomials and full boundary data for planar88/tile288/tile512 from the authors or original supplementary code. Record their provenance and checksums, then add `PaperCase` entries. Parameters alone do not specify a code.
2. **Strengthen the distance evidence for large instances.** First obtain small-weight logical witnesses for the three LP examples. Check zero syndrome, nonmembership in the stabilizer space, and weight at most the reported upper bound. This certifies the upper bound without requiring a proof of minimality.
3. **Add more expensive minimum-distance certification.** BP90/BP108, BB144, and mitten150 can use solvers with explicit optimality or unsatisfiability outcomes. On timeout, report only established bounds. Record solver versions and certificates; a heuristic minimum is not an exact distance.
4. **Treat scale as a separate performance milestone.** Retain the finite examples as correctness regressions before introducing sparse matrices, compressed orbit indices, and larger groups. The current dense implementation prioritizes auditability.
5. **Separate decoder and experimental reproduction.** Reproducing thresholds or hardware results requires specified noise models, decoders, sample counts, and original data. Construction-matrix tests do not establish these claims.

The report preserves unresolved and uncertified statuses explicitly. Future inputs and algorithms can strengthen individual claims without duplicating construction code in notebooks.
