# Implementation Specification: Illumination-Robust Classical Image-Processing Pipelines with Kartezio

> **Purpose of this document:** This is the implementation contract for Codex, Antigravity, or a developer working directly in this repository. Follow it in order. Inspect the existing repository before changing anything. Preserve Kartezio's current structure and behavior unless a minimal, justified change is necessary.

## 0. Instructions to the coding agent

1. **Inspect first; do not guess APIs.** Read `README`, `pyproject.toml`/requirements, source modules, tests, and current Git status before editing. Identify the actual Kartezio version, package layout, public APIs, graph execution flow, primitive registration flow, fitness interface, serialization, and test commands.
2. **Do not rewrite or replace Kartezio.** Keep existing `src/kartezio/` modules and names. Do not move the upstream code under `original/`, rename its files, or replace its trainer with a newly invented one.
3. **Keep project-specific code isolated.** New benchmark, metric, experiment, and analysis code belongs under `projects/illumination_robustness/`. Only make small changes to `src/kartezio/` if the current public API cannot support a required operation. Explain each such change in a comment and in `docs/implementation_decisions.md`.
4. **Work in small, testable increments.** Implement one stage, run its relevant tests, fix issues, and only then continue. Do not create all files as empty placeholders and claim completion.
5. **Never fabricate data or results.** Do not invent benchmark files, ground-truth masks, scores, runtime values, citations, or API names. Missing data or unsupported metric details must be reported as blockers.
6. **Do not claim exact reproduction without validation.** The supplied Python conversion of the MATLAB code is approximate and unvalidated. Treat it as a reference to inspect, not ground truth.
7. **Use deterministic, explicit behavior.** Fix seeds, split IDs, condition IDs, normalization settings, primitive definitions, thresholding conventions, and empty-mask behavior in configuration or protocol documentation.
8. **Do not tune on the test set.** Held-out scenes/conditions may only be used for final evaluation, never for fitness, primitive selection, normalization fitting, threshold tuning, or choosing the final pipeline.
9. **At the end of every implementation stage**, report: files changed, commands run, tests passed/failed, remaining blockers, and next stage. Do not say a stage passed if the command was not run.
10. **Ask only when truly blocked.** If a required dataset path, annotation schema, or API contract is unavailable, inspect the repo and available files first. Then create a concise blocker note in `docs/implementation_decisions.md` and ask for the missing item instead of inventing a substitute.

## 1. Project objective and research question

### Objective

Use Kartezio's Cartesian Genetic Programming (CGP) to evolve interpretable classical image-processing pipelines for segmentation, and evaluate whether robustness-aware fitness improves performance under changing illumination/shading and noise.

### Research question

> Does optimizing classical image-processing pipelines across multiple illumination/noise conditions improve segmentation robustness and generalization to held-out scenes compared with optimizing on easier conditions alone?

Treat this as a hypothesis, not an expected outcome.

### Main comparison

- **Baseline methods:** Otsu and Sobel.
- **Method A — quality-only evolution:** fitness is measured on a fixed set of easier training conditions.
- **Method B — robustness-aware evolution:** fitness is measured on a fixed set of training conditions covering easy, medium, and hard conditions.
- **Evaluation:** both evolved methods and baselines use the same documented evaluation metrics and held-out data protocol.

Do not add NSGA-II, deep learning, a second dataset, a classifier, or unrelated framework features before the core comparison works.

## 2. Benchmark and evidence constraints

Primary dataset reference:

Khan, A. u. M., Mikut, R., and Reischl, M. (2015). “A Benchmark Data Set to Evaluate the Illumination Robustness of Image Processing Algorithms for Object Segmentation and Classification.” *PLOS ONE*, 10(7), e0131098. https://doi.org/10.1371/journal.pone.0131098

Expected benchmark organization:

- 4 scenes.
- 13 shading levels.
- 14 noise levels.
- 728 images in total if all combinations are present.
- Nominal image size: 1000 × 1500 grayscale pixels.
- Filename pattern: `benchdata_{scene}_{shading}_{noise}.tif`, for example `benchdata_1_1_1.tif`.

The exact contents of the downloaded benchmark and annotation files must be inspected. Do not assume every expected annotation exists or that every annotation is a binary mask. Distinguish binary masks, instance-label images, and object metadata.

### Important metric limitation

The paper defines:
- Full quality: `Q = μ1 * μ2 * μ3`.
- Segmentation quality: `Q_seg = μ1 * μ2`.
- Full robustness for scene `r`: `R(r) = (1/13) * Σ[b=1..13] Q(r,b,b+1)` for the paper's prescribed adjacent condition pairs.

`μ1` is object-count quality, `μ2` is segmentation/overlap quality, and `μ3` is classification quality. The extracted MATLAB package did not contain `klf_en6.m` and `klf_an6.m`, which the source calls for classification. Therefore:

- The core project is segmentation-only.
- Do not report full `Q` unless the classifier and all required components are recovered and validated.
- Use the name `Q_seg` only if the exact paper formulas, fuzzy membership functions, parameter values, and required preprocessing are reproduced.
- If exact formulas/parameters cannot be recovered, use transparent supplementary metrics such as Dice, IoU, and object-count error, and label them as custom metrics.
- The project analogue `R_seg` is `mean(Q_seg)` across the same 13 prescribed adjacent condition pairs. Clearly label it as segmentation-only and adapted; do not compare it directly to the paper's full `R`.
- The paper's scene-1 full robustness values for Otsu (0.54) and Sobel (0.47) include classification. They are historical context only and are not directly comparable to Dice, IoU, or custom segmentation-only scores.

## 3. Repository structure

First inspect the actual repository. Preserve its existing files and add the following structure where compatible. If the project already has a conventional source layout or test directory, adapt this plan minimally and document the mapping.

```text
repo-root/
├── src/
│   └── kartezio/                       # Preserve current framework code and names
│       └── ...                         # Existing CGP, graph, primitive, trainer, reader, utility modules
├── projects/
│   └── illumination_robustness/
│       ├── __init__.py
│       ├── README.md                   # Project overview and commands
│       ├── config/
│       │   ├── baseline.yaml
│       │   ├── quality_only.yaml
│       │   └── robustness_aware.yaml
│       ├── data/
│       │   ├── __init__.py
│       │   ├── manifest.py              # Scan files, parse IDs, validate coverage
│       │   ├── loader.py                # Load image/annotation without silent rescaling
│       │   ├── ground_truth.py           # Normalize annotation representation, validate alignment
│       │   └── splits.py                 # Deterministic train/test scene and condition splits
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── normalization.py          # Explicit normalization policy
│       │   └── postprocessing.py         # Shared, documented mask postprocessing
│       ├── primitives/
│       │   ├── __init__.py
│       │   ├── image_ops.py              # Intensity and filtering primitives
│       │   ├── morphology.py             # Morphology primitives
│       │   └── registry.py               # Register project primitives using actual Kartezio API
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py                # Dice, IoU, custom count metrics; paper metrics only if verified
│       │   ├── object_metrics.py          # Connected components and object-count rules
│       │   ├── evaluator.py              # Execute pipeline and score a sample
│       │   ├── robustness.py              # Adjacent-pair curve and R_seg
│       │   └── baselines.py               # Otsu and Sobel reference pipelines
│       ├── evolution/
│       │   ├── __init__.py
│       │   ├── fitness.py                # Fixed training-set aggregation
│       │   ├── dataset_sampler.py         # Fixed, stratified condition selection
│       │   └── run_experiment.py          # Adapter to actual Kartezio training API
│       └── analysis/
│           ├── __init__.py
│           ├── aggregate_results.py       # Per-image → per-run/scene summaries
│           ├── statistics.py              # Descriptive statistics and uncertainty
│           └── plots.py                   # Figures saved to results/
├── scripts/
│   ├── build_manifest.py
│   ├── validate_data.py
│   ├── run_baselines.py
│   ├── run_evolution.py
│   ├── evaluate_saved_pipelines.py
│   └── summarize_results.py
├── tests/
│   ├── ...                                # Preserve existing Kartezio tests
│   └── test_illumination_robustness/
│       ├── test_manifest.py
│       ├── test_ground_truth.py
│       ├── test_metrics.py
│       ├── test_preprocessing.py
│       ├── test_primitives.py
│       ├── test_splits.py
│       ├── test_fitness.py
│       └── test_reproducibility.py
├── docs/
│   ├── benchmark_protocol.md              # Frozen protocol and all deviations
│   └── implementation_decisions.md        # Decisions, blockers, deviations, upstream patches
├── data/
│   ├── raw/                               # User-provided original benchmark files; not committed
│   ├── interim/
│   ├── processed/
│   └── manifests/
├── results/                               # Generated artifacts; not committed by default
├── pyproject.toml                         # Preserve; make minimal dependency/config changes
├── .gitignore
└── README.md
```

### Code boundaries

- `src/kartezio/`: framework logic only.
- `projects/illumination_robustness/`: benchmark and research logic.
- `scripts/`: thin command-line entry points that call project modules; avoid duplicating algorithms in scripts.
- `tests/`: preserve upstream tests and add separate project tests.
- `data/`: source data and generated intermediate data.
- `results/`: experiment outputs, logs, plots, pipeline artifacts, and summary tables.

Do not put project-specific metric formulas or benchmark-specific assumptions into generic Kartezio modules.

## 4. Output and artifact policy — where to save everything

Every run must create a unique run directory under `results/`. Never overwrite a previous run by default.

Use a structure similar to:

```text
results/
└── YYYYMMDD_HHMMSS_<experiment>_seed-<seed>/
    ├── config.yaml                    # Exact resolved config, not just a path to input YAML
    ├── metadata.json                  # Git commit, Python/package versions, OS, seed, timestamps
    ├── manifest_snapshot.csv          # Exact images used and IDs
    ├── split.json                     # Train/test scenes, conditions, and image IDs
    ├── logs/
    │   ├── run.log
    │   └── errors.jsonl               # One JSON object per failed sample/graph
    ├── metrics/
    │   ├── per_image.csv              # One row per image per method/pipeline/run
    │   ├── per_scene.csv
    │   ├── per_condition.csv
    │   ├── per_run_summary.json
    │   └── fitness_history.csv
    ├── pipelines/
    │   ├── best_pipeline.json         # Only if supported by Kartezio's real serializer
    │   ├── best_pipeline.txt          # Human-readable graph/operation listing
    │   └── pipeline_metadata.json
    ├── predictions/
    │   └── <pipeline_id>/<image_id>.png
    ├── figures/
    │   ├── quality_vs_shading.png
    │   ├── quality_vs_noise.png
    │   ├── robustness_by_scene.png
    │   └── fitness_history.png
    └── report.md
```

Rules:
1. The run ID should be unique; include UTC timestamp or collision-safe suffix, experiment name, and seed.
2. Save the resolved config, all split IDs, and manifest snapshot for every run.
3. Save **raw per-image results first**. Aggregated results must be regenerable from per-image CSVs.
4. Save failed evaluations in `errors.jsonl` with run ID, pipeline ID, image ID, exception type, and concise message. Do not silently drop failures.
5. Save predicted masks only when requested or for a defined audit subset to avoid unnecessary storage. Always save masks for the best selected pipeline on the held-out audit set.
6. Do not save only screenshots or plots; save machine-readable CSV/JSON as well.
7. Store result paths and metadata in the final `report.md`.
8. Do not commit large results, raw benchmark images, caches, virtual environments, or generated masks by default. Include a small sample result only if it is synthetic or redistribution is allowed.
9. The aggregation script must be runnable later without rerunning evolution.

### Recommended `.gitignore` additions

Inspect the current `.gitignore` and append only missing patterns:

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/

# Local benchmark data
data/raw/*
data/interim/*
data/processed/*
!data/**/.gitkeep

# Generated experiment outputs
results/*
!results/.gitkeep

# Logs and local environments
*.log
.env
```

Do not ignore source configs, code, docs, or test fixtures. Do not overwrite an existing `.gitignore` wholesale.

## 5. Dataset manifest and loading requirements

### Manifest builder

Implement `projects/illumination_robustness/data/manifest.py` and `scripts/build_manifest.py`.

For every discovered image:
- Parse `scene`, `shading`, and `noise` from the filename using a strict regular expression.
- Record absolute or repository-relative path consistently; prefer repository-relative paths in saved manifests when possible.
- Record width, height, dtype, channel count, and checksum.
- Locate matching annotations only using a documented naming rule or explicit mapping file.
- Detect duplicate `(scene, shading, noise)` keys, unparseable names, missing expected tuples, missing annotations, and unreadable files.
- Produce `data/manifests/benchmark_manifest.csv` plus a validation summary.
- Do not create fake annotations or silently omit bad files.

### Loader

- Load TIFF image data while preserving original numeric values and dtype until the configured normalization stage.
- Reject unexpected color images unless an explicit conversion policy is configured.
- Check image/annotation shape and spatial alignment.
- Return a structured record containing image array, annotation array, scene/shading/noise IDs, and stable `image_id`.
- Raise clear, actionable errors for missing or malformed files.

### Ground-truth interpretation

Inspect actual files before choosing behavior:
- Binary mask: foreground/background only.
- Instance-label image: each object has a distinct nonzero ID.
- MATLAB `.mat` structures: inspect keys, shapes, and types; do not assume a variable name.
- Object metadata: document whether counts are provided directly or inferred from labels.

Keep the original annotation intact. Convert to a binary mask only in an explicit function. If an instance map is reduced to binary foreground, retain the instance map for object-count evaluation.

## 6. Preprocessing and baseline protocol

### Normalization

Inspect `im_norm.m`, the benchmark macro, and its actual parameter values before implementing a reproduction. The source has multiple modes (including `musig`, `musig+prctile`, `prctile`, and `minmax`) and pixel-reversal behavior. Do not choose a mode by guesswork.

Implement a normalization function with:
- explicit mode and parameters;
- stable input/output dtype and range contract;
- explicit foreground polarity/pixel-reversal policy;
- tests for constant images, clipping, NaN/Inf, and expected output range.

If exact macro parameters are unavailable, choose one defensible Python normalization protocol, record it as a project-defined protocol, and do not call the result exact reproduction of the MATLAB benchmark.

### Baseline methods

Implement Otsu and Sobel in `evaluation/baselines.py`. Each baseline should expose a function that accepts one image plus an explicit config and returns a binary prediction mask. It must not read global configuration implicitly.

- **Otsu:** thresholding, inversion fallback if adopted, border clearing, hole filling, morphology, and small-object filtering must be explicit and ordered.
- **Sobel:** specify how gradient magnitude is computed, how a threshold is selected, and all postprocessing.
- Do not describe Python Sobel as MATLAB-equivalent until masks and thresholds have been compared on a small representative set.
- Baselines and evolved pipelines must use the same evaluation masks and metrics. If normalization differs by design, report it clearly.

The inspected MATLAB source contains suspicious region-area indexing and a potentially incorrect class-count expression. Do not silently reproduce suspicious expressions as intended metric definitions. If a legacy mode is needed for comparison, isolate it, name it, and test it.

## 7. Metric definitions and equations

### Binary overlap

For predicted foreground pixels `P` and ground-truth foreground pixels `G`:

- `TP = |P ∩ G|`
- `FP = |P \ G|`
- `FN = |G \ P|`

```text
Dice = 2*TP / (2*TP + FP + FN)
IoU  = TP / (TP + FP + FN)
```

Define empty-mask behavior in `docs/benchmark_protocol.md`. Recommended convention:
- both masks empty → score `1`;
- exactly one mask empty → score `0`.

Unit tests must cover both cases.

### Supplementary object-count metric

If valid instance annotations exist:

```text
N_gt   = number of ground-truth objects
N_pred = number of predicted objects
CountError = abs(N_pred - N_gt) / max(N_gt, 1)
CustomCountQuality = max(0, 1 - CountError)
```

Use a documented connectivity rule (4-connected or 8-connected in 2-D). If instances touch, connected-component counting may merge them; state this limitation. Label this metric as custom unless it exactly matches the paper's definition.

### Paper-defined segmentation score

Only implement paper `Q_seg = μ1 * μ2` after recovering and validating the exact definitions of `μ1`, `μ2`, their fuzzy membership functions, parameters, and required preprocessing from the paper/source. Add a source note for every formula and parameter. If this cannot be completed, do not use the name `Q_seg` for a custom score; use a name such as `custom_segmentation_score`, and use Dice/IoU as the primary verified metrics.

### Robustness across prescribed adjacent pairs

The paper's full robustness equation is:

```text
R(scene) = (1/13) * sum_{b=1..13} Q(scene, b, b+1)
```

For segmentation-only evaluation, only when the exact paper segmentation score is implemented:

```text
R_seg(scene) = (1/13) * sum_{b=1..13} Q_seg(scene, b, b+1)
```

The pair indexing must match the benchmark paper's prescribed adjacent-condition sequence; verify how shading/noise indices map to each pair before coding. Do not infer pairs from alphabetical file order. If using Dice/IoU instead, define a clearly named analogue such as `R_dice` or `R_iou`, and document that it is a project-defined curve summary.

### Evolutionary fitness

Use a fixed training image set for all individuals within one run.

```text
F_quality(P) = mean(metric(P, I_i) for I_i in D_easy)
F_robust(P) = mean(metric(P, I_i) for I_i in D_multi)
```

`D_easy` is a predeclared training subset of easier conditions. `D_multi` is a predeclared, stratified set covering easy, medium, and hard training conditions. Use equal weighting initially.

If the current Kartezio trainer minimizes fitness, transform the maximization score with:

```text
loss(P) = 1 - F(P)
```

If it maximizes, pass `F(P)` directly. Inspect the actual trainer and write a unit test proving that a better segmentation receives better selection fitness. Do not assume objective direction.

If adding condition-level weighting later, define:

```text
F_weighted(P) = sum_k w_k * mean_metric(P, D_k)
where w_k >= 0 and sum_k w_k = 1
```

Do not introduce weights until the unweighted experiment works; freeze any weights before held-out evaluation.

## 8. Experimental splits and leakage prevention

Use leave-one-scene-out evaluation:

- Train scenes 2, 3, 4 → test scene 1.
- Train scenes 1, 3, 4 → test scene 2.
- Train scenes 1, 2, 4 → test scene 3.
- Train scenes 1, 2, 3 → test scene 4.

Within each rotation:
- The test scene must not appear in fitness evaluation, primitive selection, configuration selection, or final pipeline selection.
- Use identical held-out data for quality-only and robustness-aware methods.
- The quality-only method uses a predeclared easy-condition subset from training scenes.
- The robustness-aware method uses a predeclared stratified subset from training scenes.
- Record every training and test image ID in `split.json`.
- Do not define “easy” by looking at results. Use benchmark condition indices and document the rule.
- Run at least 3 seeds per strategy/rotation; target 5 if runtime permits.
- Do a small smoke experiment before the full experiment. Mark smoke results as development results, not final evaluation.

The paper's adjacent-pair robustness score and leave-one-scene-out generalization answer different questions. Keep them separate in tables and plots.

## 9. Kartezio integration and primitives

### Integration steps

1. Run the existing repository tests and at least one unmodified example before making changes.
2. Trace the actual path from primitive registration → graph construction → graph execution → fitness evaluation → selection → serialization.
3. Implement a project evaluator/adapter using the real API. Do not create a parallel CGP engine.
4. Create a one-input grayscale → one-output mask smoke pipeline.
5. Register project primitives through the existing public registration mechanism.
6. Run one short evolutionary run on a tiny, fixed synthetic/sample set.
7. Test save/load or the framework's actual serialization path; if no serializer exists, document the minimal custom artifact format rather than inventing one inside core modules.
8. Run the two fitness strategies with identical evolution settings except the condition sampling/fitness strategy.

### Initial primitive set

Keep the first primitive set deliberately small:

- Intensity: identity and complement.
- Filtering: Gaussian blur and median filter with a small set of fixed kernel sizes.
- Thresholding: Otsu.
- Edges: Sobel magnitude or binary Sobel.
- Morphology: erosion, dilation, opening, closing, hole filling, border clearing, and small-component removal.

Only include operations that can be executed consistently on the benchmark images and whose behavior is tested. Avoid an excessively large parameter search. Every primitive must declare:
- name and purpose;
- input type/shape;
- output type/shape;
- parameters and legal bounds;
- behavior on constant images or invalid inputs;
- deterministic behavior;
- source/reference if its behavior is based on a published implementation.

### Graph failure behavior

- Catch failures at the evaluator boundary, not by scattering broad `try/except` blocks across primitives.
- Record exception type and message in `errors.jsonl`.
- Assign a deterministic worst fitness to an invalid graph according to the trainer's objective direction.
- Keep invalid-graph penalty separate from valid empty-mask predictions.
- Do not silently convert malformed outputs into plausible masks.

## 10. Comments, docstrings, naming, and code style

The code must be understandable to a new team member who has not read the full project history.

### Comments

- Explain **why** a non-obvious choice exists, not what the next line literally does.
- Add comments for deviations from the paper, MATLAB/Python behavior differences, foreground polarity, empty-mask policy, metric assumptions, and any upstream compatibility patch.
- Never leave speculative comments such as `# this should work`, `# probably correct`, or claims that a function is exact when it is not validated.
- Use `TODO:` only for an explicit unresolved task. Each TODO must name the missing condition or decision; do not use TODO as a substitute for implementing required work.
- Do not comment out large blocks of obsolete code. Remove unused code or preserve historical alternatives in version control.
- If a formula comes from the paper, cite the paper and the relevant equation/section in a docstring or `docs/benchmark_protocol.md`.

### Docstrings

Use Google-style docstrings for public functions and classes. Include:
- what the function/class does;
- parameter meaning and expected types/shapes;
- return type and semantics;
- important assumptions, side effects, and exceptions;
- metric reference or protocol name where relevant.

Example:

```python
def dice_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    \"\"\"Compute binary Dice overlap.

    Args:
        prediction: 2-D array; nonzero values are treated as foreground.
        ground_truth: 2-D array aligned with prediction; nonzero values are foreground.

    Returns:
        Dice score in [0, 1]. Empty-mask behavior follows benchmark_protocol.md.

    Raises:
        ValueError: If shapes differ or inputs are not 2-D.
    \"\"\"
```

### Naming and typing

- Use descriptive `snake_case` for functions/variables and `PascalCase` for classes.
- Add type hints to public functions and nontrivial internal interfaces.
- Avoid unexplained one-letter names except in formulas where standard notation is documented.
- Avoid magic numbers: place thresholds, kernel sizes, area cutoffs, and condition lists in configuration or named constants with a source note.
- Do not use global mutable experiment state.
- Keep scripts thin; put reusable logic in modules.
- Use Python's standard `logging` module instead of print statements for run progress. `print` is acceptable for concise CLI summaries only.
- Use `pathlib.Path` for paths and explicit UTF-8 for text/JSON.
- Use a consistent formatter/linter already present in the repository. If none exists, do not add a large tooling stack; add a minimal formatter/linter only if practical.
- Avoid unexplained broad exception handling. Catch known errors at well-defined boundaries and preserve tracebacks in logs when appropriate.

## 11. Configuration requirements

Keep experiment choices in YAML/JSON configuration, not scattered constants. Example schema (adapt to existing dependency choices):

```yaml
experiment:
  name: robustness_aware
  seed: 42

dataset:
  manifest_path: data/manifests/benchmark_manifest.csv

protocol:
  metric: dice
  normalization:
    mode: project_defined
    parameters: {}
  empty_mask_policy: both_empty_one
  connectivity: 2d_8_connected

split:
  strategy: leave_one_scene_out
  test_scene: 1
  train_scenes: [2, 3, 4]

training_conditions:
  strategy: stratified_multi_condition
  condition_ids: []  # Must be populated and frozen before final runs.

evolution:
  primitive_set_version: v1
  population_size: null  # Set only after timing a smoke run.
  generations: null
  evaluation_image_count: null

output:
  root: results/
  save_all_predictions: false
  save_best_test_predictions: true
```

The `null` values are deliberate placeholders. The coding agent must not invent “optimal” values. First time a small smoke run, then choose practical values, record the rationale, and freeze them before final runs. Config validation should fail early if required values are unset.

## 12. Result schemas and analysis

### Per-image CSV

At minimum, write one row per image evaluated:

```text
run_id,method,seed,split_id,scene,shading,noise,image_id,
pipeline_id,metric_protocol,dice,iou,custom_count_error,
gt_object_count,predicted_object_count,status,error_type,runtime_ms
```

Only include columns that have defined semantics. Use blank/NaN for genuinely unavailable metrics; do not use zero to mean “not computed.” Keep metric names explicit; do not put a custom metric in a column named `qseg`.

### Per-condition and per-scene aggregation

- Aggregate from per-image CSV, never from rounded display values.
- Report mean, standard deviation, number of images, and number of failed evaluations.
- Keep per-scene results visible; a single pooled mean can hide scene-specific failure.
- For repeated seeds, report mean and spread across seeds. If statistical tests are used, predefine them and do not overclaim from a small number of scenes.
- Preserve unrounded values in CSV/JSON; round only in final report tables.
- Plot quality against shading and noise condition separately where meaningful.
- Label every plot with metric name, aggregation level, method, and held-out/training status.
- Save plots as PNG (for report insertion) and optionally PDF/SVG if vector output is useful.

### Report outputs

Generate `results/<run_id>/report.md` and an aggregate report under `results/summary/` containing:
- experiment purpose and protocol;
- data coverage and exclusions;
- exact metric definitions;
- baseline and evolved method results;
- split/seed information;
- plots and their paths;
- failures and deviations;
- limitations and conclusions that are directly supported by the results.

## 13. Testing requirements

Preserve all existing Kartezio tests. Add tests for:

- Manifest filename parsing, duplicate keys, missing combinations, and invalid filenames.
- Ground-truth shape alignment and label interpretation.
- Dice/IoU perfect match, disjoint masks, partial overlap, both-empty masks, and one-empty masks.
- Object count and connectivity behavior.
- Normalization on regular, constant, low-dynamic-range, and invalid arrays.
- Each primitive's output shape, dtype, deterministic behavior, and synthetic expected result.
- Fixed train/test splits and absence of leakage.
- Fitness aggregation and maximization/minimization direction.
- Invalid graph handling versus legitimate empty predictions.
- Serialization/reload and deterministic output, if supported by the actual Kartezio API.
- Result writing: one row per evaluated image, stable columns, and no overwrite by default.

Run relevant tests after each stage, then run the full test suite before finalizing. Record actual commands and outcomes in `docs/implementation_decisions.md` or the run report.

## 14. Implementation sequence and stage gates

### Stage 0 — Repository reconnaissance
- Inspect repository tree, Git status, dependency files, current tests, examples, and public Kartezio APIs.
- Run baseline tests/examples without modifications.
- Record the actual API flow and test command.
- **Gate:** do not proceed until framework status and blockers are recorded.

### Stage 1 — Data and manifest
- Implement manifest creation, validation, loading, and ground-truth inspection.
- Create a data audit report.
- **Gate:** expected coverage is verified or every missing item is documented; masks/annotations are visually inspected.

### Stage 2 — Metrics and protocol
- Implement Dice/IoU and custom count metrics with tests.
- Implement paper `Q_seg` only if exact formulas and parameters are verified.
- Write `docs/benchmark_protocol.md`.
- **Gate:** formulas, empty-mask behavior, foreground convention, and condition-pair mapping are frozen.

### Stage 3 — Baselines
- Implement fixed Otsu and Sobel pipelines.
- Compare to MATLAB on representative images if MATLAB is available.
- Run sample first, then full planned baseline evaluation.
- **Gate:** masks and scores are reproducible and protocol differences are documented.

### Stage 4 — Kartezio adapter and primitives
- Register minimal primitives using the actual framework API.
- Build and test one complete graph.
- Add evaluator-to-fitness adapter.
- **Gate:** synthetic/sample graph evaluates correctly and serialization/reload works or its limitation is documented.

### Stage 5 — Fitness strategies
- Implement quality-only and robustness-aware fitness over fixed condition sets.
- Verify objective direction.
- **Gate:** a hand-constructed set of sample scores yields expected fitness values.

### Stage 6 — Smoke evolution
- Run a small, cheap evolution.
- Inspect fitness history, graph validity, saved artifacts, and reproducibility.
- **Gate:** end-to-end run completes and can be re-evaluated from saved artifacts.

### Stage 7 — Final experiments
- Freeze configs, split IDs, primitive set, metric protocol, and seeds.
- Run leave-one-scene-out rotations and repeated seeds within compute limits.
- Evaluate held-out scenes only after pipeline selection is frozen.
- **Gate:** raw results, logs, configs, metadata, and pipeline artifacts are saved.

### Stage 8 — Analysis and documentation
- Recompute all summaries from raw per-image CSVs.
- Generate figures and report.
- Review for leakage, unsupported claims, missing run metadata, and metric naming errors.
- **Gate:** another team member can reproduce a run from the instructions and saved config.

## 15. Five-week schedule

- **Week 1:** repository reconnaissance, benchmark acquisition/inspection, manifest, data audit.
- **Week 2:** metrics, protocol documentation, Otsu/Sobel baselines and validation.
- **Week 3:** Kartezio adapter, primitives, fitness strategies, smoke evolution.
- **Week 4:** freeze experiment protocol; run the planned seeds and scene-held-out rotations.
- **Week 5:** final evaluation, plots, error analysis, reproducibility checks, and report.

If the benchmark annotations or exact paper metrics are blocked, do not stall all progress: implement and test Dice/IoU as clearly named supplementary metrics while documenting that paper-score reproduction is incomplete. Do not claim the core scientific comparison is complete until ground truth and held-out evaluation work.

## 16. Acceptance criteria

The implementation is complete only when:

1. Existing Kartezio code remains in place and its tests are preserved.
2. New project code is isolated under `projects/illumination_robustness/`.
3. Dataset manifest and ground-truth schema are validated.
4. Metric formulas and preprocessing policies are documented and tested.
5. Otsu and Sobel baselines run reproducibly.
6. Both fitness strategies run end-to-end through Kartezio.
7. Both strategies are evaluated on identical held-out data.
8. Repeated seeds and scene-held-out splits are run, or compute limitations are clearly documented.
9. Raw per-image CSVs, resolved configs, split files, logs, metadata, pipeline artifacts, and plots are saved under unique run directories.
10. Tests and actual command outcomes are recorded.
11. No test leakage, fabricated values, or unsupported claims are present.
12. README instructions allow another developer to reproduce the experiment.

## 17. Final instructions to the coding agent

Implement the project in stages, not as a single giant patch. Start by inspecting the repository and reporting the existing layout, actual Kartezio API flow, available data/annotations, and test status. Then implement Stage 1 only. Do not start evolutionary training until the manifest, ground truth, metric tests, and baseline pipeline are valid. After each stage, report changed files, commands executed, test outcomes, saved artifact paths, and unresolved blockers.
