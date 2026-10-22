# Implementation Decisions, Blockers and Deviations

This file records every decision, deviation, blocker and upstream patch for the
illumination-robustness project. Entries are append-only; superseded decisions are
marked as such rather than deleted.

Primary sources (local copies in `papers/`, not committed):

- **[P]** Khan, Mikut, Reischl (2015), PLOS ONE 10(7): e0131098 — main paper.
- **[S1]** S1 Text — data and object information (Table A).
- **[S2]** S2 Text — data labelling.
- **[S3]** S3 Text — artifact level.
- **[S4]** S4 Text — quality measures for segmentation and classification.
- **[M]** Released MATLAB code in `hardware_items/code/` (`autosegbench.m`,
  `im_norm.m`, `script_benchdata_artefact.makrog`).

**Precedence rule (user instruction, 2026-10-19):** benchmark calculations and
measures follow the paper and supplements [P, S1–S4]. The MATLAB code [M] is used
only where the paper is silent (e.g. exact baseline postprocessing parameters), and
every such use is labelled. Scope is **segmentation only**: µ3 / classification is
not implemented.

---

## Stage 0 — Repository reconnaissance (2026-10-19)

### Repository

| Item | Finding |
|---|---|
| Package | `kartezio` 1.0.2, `src/kartezio/`, build backend hatchling |
| Python | `>=3.11`; local interpreter CPython 3.12.6 via `uv` 0.12.10 |
| Test command | `$env:UV_LINK_MODE="copy"; uv run pytest -q` |
| Upstream test status (unmodified) | **39 passed, 1 failed** — `tests/components/test_reader.py::ReaderTestCase::test_RoiPolyHedron`. Pre-existing; not caused by this project; not investigated further. |
| Git | HEAD `4f9bf6a`; `.gitignore` locally modified by user (`hardware_items/`, `papers/`) |

**Environment note.** `uv run` fails on this machine with
`os error 396 (incompatible hardlinks)` because the workspace is on a cloud-synced
Desktop. Setting `UV_LINK_MODE=copy` resolves it.

### Kartezio API flow (as read from source; verified during the pilot test)

- Primitives/endpoints/fitness are registered with the `@register(<Fundamental>)`
  decorator in `kartezio.core.components`.
- `KartezioTrainer(n_inputs, n_nodes, libraries, endpoint, fitness)` →
  `.fit(n_generations, train_x, train_y, callbacks=...)` →
  `.evaluate(x, y)`; `.print_python_class(name)` exports a graph.
- **Objective direction:** fitness classes return a *loss*; e.g.
  `kartezio.core.fitness.IoU.evaluate` returns `1 - IoU`. The trainer therefore
  **minimises**. Project fitness will be supplied as `loss = 1 - F`. A unit test in
  Stage 5 must prove a better segmentation receives a lower loss.

### Data location

**Decision (user instruction):** benchmark data is read in place from
`hardware_items/` (already gitignored). The spec's `data/raw/` is not used. Generated
manifests go to `data/manifests/`; generated audits/results go to `results/`.

### Benchmark contents (inspected)

| Path | Contents |
|---|---|
| `hardware_items/data/benchdata_RR_BB_NN.tif` | 728 files, all `(r,b,n)` ∈ 4×13×14; uint8, 1000×1500, single channel |
| `hardware_items/truth/benchdata_{r}_gd.mat` | variable `ground_truth`, uint8 {0,1}, **3456×5184** (camera resolution) for all scenes |
| `hardware_items/truth/benchdata_{r}_class.mat` | variable `im_label`, uint8 class map {0..K}; 3456×5184 for r=1, 1000×1500 for r=2..4 |
| `hardware_items/truth/benchdata_0r_01_01.tif` | byte-identical statistics to the corresponding data image — **not** masks |
| `hardware_items/truth/benchdata_{r}_gd.jpg` | RGB visualisation; not used |

Ground truth is **per scene**, shared across all 182 shading/noise conditions of
that scene, because object positions were fixed during acquisition [P, p.4].
Filenames are zero-padded (`benchdata_02_01_03.tif`), as stated in [P, p.3]; the
implementation spec's `benchdata_1_1_1.tif` example is incorrect.

---

## Paper-derived decisions (2026-10-19)

### D1. Robustness image series and normalisation of R

[P, Eq. 3]: `R = (1/B) Σ_{b=1..B} Q(r, b, b+1)`, B = 13, using the image series
`X(r, k, k+1)`, k = 1..13 — i.e. **13 single images** with shading `b` and noise
`b+1`, not pairs of images.

- The implementation spec's description "adjacent condition pairs" is corrected to
  this definition.
- [M] (`script_benchdata_artefact.makrog`) additionally includes image `(1,1)` and
  divides by 14. **Not followed**; the paper definition is used.
- Project analogue: `R_seg(r) = (1/13) Σ_{b=1..13} Q_seg(r, b, b+1)`.

### D2. Fuzzy membership function

[S3, Eq. 2] defines a monotonically *decreasing* spline function
`µ(θ; s, α, β)` (1 below α, 0 above β). With s = 2 this equals MATLAB `zmf(θ,[α β])`.

### D3. Direction of µ1 and µ2 — internal inconsistency in S4 (**flagged**)

[S4, Eqs. 3, 5] write `µ1 = 1 − µ(q1; 2, 0, 1)` and `µ2 = 1 − µ(q2; 2, 0, 1)`, while
[S4, Eq. 8] writes `µ3 = µ(q3; 2, 0, 1)`. Since q1, q2 are *error* measures (0 =
perfect) and µ is decreasing, the literal `1 − µ` form gives µ = 0 for a perfect
segmentation, contradicting [P, p.6] ("larger values show high robustness") and
[M] (`y1_hat = evalmf(q1_hat, [0 a], 'zmf')`, no `1 −`).

**Decision:** `µ1 = µ(q1; 2, 0, 1)`, `µ2 = µ(q2; 2, 0, 1)` (same form as µ3). This is
the only reading consistent with the paper's stated semantics. Documented as a
correction of a typographical inconsistency, not a free design choice.

### D4. Fuzzy parameters

Paper parameters are used: s = 2, α = 0, β = 1 for both µ1 and µ2 [S4].
[M] uses `apriori1 = 50` for µ1, which makes µ1 ≈ 1 for any relative count error in
[0, 1]. **Not followed.**

### D5. Segmentation measure 2 (q2)

[S4, Eq. 4]: `q2 = Σ|sign(δtruth − δseg)| / Σ sign(δtruth + δseg)` =
`|G XOR P| / |G ∪ P|` = `1 − IoU(G, P)`.
[M] divides by `max(|G|, |P|)` instead. **Not followed.**
Empty-union case (both masks empty, 0/0): defined in `docs/benchmark_protocol.md`.
For this benchmark G is never empty, so the case can only arise in synthetic tests.

### D6. Segmentation measure 1 (q1) — matching rule (**flagged ambiguity**)

[S4] text says a predicted BLOB S_u matches if its maximum overlap "exceeds 1/3 of
T_v", but [S4, Eq. 1] writes `max_z card(S_u ∩ T_z) > (1/3)·C_u`, where C_u is the
pixel count of **S_u**. [M] compares against `0.33 * im_otsu(feat_ot(m).Area)`,
which indexes a pixel *value* by the area (an indexing bug), so [M] cannot arbitrate.

**Decision:** follow the equation (threshold relative to the predicted BLOB size
C_u). It also penalises under-segmentation (one huge BLOB covering many objects fails
the test). The alternative reading is kept as an explicit, non-default option
(`overlap_reference="ground_truth_blob"`) for sensitivity analysis only.

Other q1 details from [S4]:
- BLOBs S_u are processed sequentially; on a match, **all** remaining T_v that
  overlap S_u are removed (S4 case "one S_u overlaps two T_v"). Non-matching S_u
  remove nothing.
- Processing order: MATLAB `regionprops` order (column-major first-pixel order), to
  be faithful to the reference implementation's sequential semantics.
- `q1 = |card(T) − card(S_c)| / card(T)`.

### D7. Ground-truth BLOBs T and object count V

[S4] defines T as the set of BLOBs (connected components) in X_truth and V = card(T).
Connectivity: 8-connected (MATLAB `bwlabel` default). Audit on the resized mask
(see D8):

| Scene | 8-conn BLOBs in GT | [S1] Table A objects | Note |
|---|---|---|---|
| 1 | 27 | 27 | match |
| 2 | 32 | 32 | match |
| 3 | 51 | 52 | one BLOB = hex nut (137 px) touching another object (2341 px) |
| 4 | 58 | 58 | match in total; class map shows one touching pair and is otherwise inconsistent with Table A per class |

**Decision:** use the paper's BLOB definition literally (V = card(T) from the binary
GT). The scene-3 discrepancy against Table A (51 vs 52) is reported, not patched.

### D8. Ground-truth preprocessing

- The paper does not specify GT preprocessing. Masks are stored at camera resolution
  (3456×5184) and must be brought to image resolution (1000×1500); both axes have the
  same scale factor 3.456.
- **Resize:** OpenCV `INTER_AREA` (area averaging) on the {0,1} mask, then
  foreground = value ≥ 0.5. [M] uses MATLAB `imresize` (bicubic, antialiased, uint8
  rounding, then `> 0`), which is effectively the same 0.5 rule on a different
  kernel; results may differ at boundary pixels. Not claimed MATLAB-identical.
- **No morphological opening.** [M] applies `imopen(ground_truth, strel('disk',3))`;
  this is not in the paper and the audit shows it splits thin objects (8-conn BLOB
  counts become 31/37/58/62 instead of 27/32/51/58). Not followed.
- The class map (`im_label`) is retained, resized with nearest neighbour when needed,
  for auditing only. Class labels in the files are a permutation of Table A's column
  order (object sizes match Table A min/max exactly).

### D9. Image normalisation

[P, p.5]: "a grayscale image is first normalized using 2% and 98% quantiles."
[M] runs `norm_type='minmax'` with `pixrev=true`, which ignores the quantiles and
only computes `1 − x/255`. **Not followed.**

**Decision (paper):** clip to the 2nd/98th percentiles (Hazen definition, matching
MATLAB `prctile`), rescale linearly to [0, 1], and invert polarity so that objects
(dark on bright background [S3]) become bright foreground. Constant-image behaviour is
defined in the protocol. To be implemented in Stage 2/3.

### D10. Baseline postprocessing

[P, Fig. 3] names hole filling, opening and border clearing but gives no parameters.
Exact steps and parameters (inversion fallback, 50 px pre-filter, 7×7 dilation,
disk-5 opening, `min_area = 100`) are therefore taken from [M] and labelled as such.
The `regionprops`/classification parts of [M] are out of scope. Stage 3.

### D11. Dependencies

`scipy` (for `.mat`), `tifffile` and OpenCV are already available (scikit-image /
opencv-python dependencies). `pyyaml` is **not** installed; it will be added as a
minimal dependency in the stage that introduces YAML configs.

### D12. Baseline validation against Paper Fig. 4

- Validated Python Otsu and Sobel baselines against Figure 4 of [P].
- Replicated the 13-image degradation series for Scene 1 ($r=1, b=1..13, n=b+1$).
- Evaluated the 5 milestone conditions indicated by arrows in Fig. 4:
  P1 ($A=0.005$), P4 ($A=0.256$), P6 ($A=0.555$), P8 ($A=0.749$), P13 ($A=1.000$).
- Failure modes match paper exactly: Otsu suffers catastrophic shadow streak false-positives
  at $A \approx 0.75$; Sobel edges fragment and disappear into empty masks at $A \ge 0.75$.
- Robustness scores without omitted classifier $\mu_3$: $R_{\text{otsu}} = 0.39$, $R_{\text{sobel}} = 0.41$.
- Detailed report: `docs/baseline_reproduction_report.md`.
- Side-by-side figure: `docs/audit_vis/paper_fig4_vs_python_comparison.png`.

---

## Stage log

| Date | Stage | Commands | Outcome |
|---|---|---|---|
| 2026-10-19 | 0 | `uv run pytest -q` (with `UV_LINK_MODE=copy`) | 39 passed, 1 failed (pre-existing `test_RoiPolyHedron`) |
| 2026-10-19 | 1 | `uv run python scripts/build_manifest.py`<br>`uv run python scripts/validate_data.py`<br>`uv run pytest tests/test_illumination_robustness -v` | 728/728 images validated and manifested; 4 scene ground truths aligned; data audit report generated; 11/11 tests passed. |
| 2026-10-19 | 2 | `uv run pytest tests/test_illumination_robustness -v` | Implemented normalization, postprocessing, Dice/IoU, paper Q_seg (with S4 sequential BLOB matching), artifact levels, and robustness indices; benchmark protocol frozen; 28/28 tests passed. |
| 2026-10-19 | 3 | `uv run pytest tests/test_illumination_robustness/test_baselines.py -v`<br>`uv run python scripts/run_baselines.py --mode smoke`<br>`uv run python scripts/visualize_paper_fig4_comparison.py`<br>`uv run python scripts/run_baselines.py --mode full` | Implemented Otsu & Sobel baseline pipelines; validated against Fig 4 of the paper; executed full 728-image benchmark (1,456 evaluations, run_id `baselines_full_20261019_182719`); generated `docs/baseline_reproduction_report.md` and `docs/audit_vis/paper_fig4_vs_python_comparison.png`; 32/32 tests passed. |

