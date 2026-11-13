# Final Experiment Handoff

This document defines the remaining experiments for the illumination-robustness study. It is the working protocol for implementation, execution, and result storage. Runtime benchmarking is intentionally excluded for now.

## Fixed settings

All runs use the same implementation and evaluation protocol:

- Seeds: `7`, `123`, and `31415`.
- Population size: `16` total candidates. In Kartezio this should be configured with `trainer.set_n_children(16)` so the strategy maintains one parent and 15 offspring.
- Generations: `500` for every primary run.
- Graph capacity: `8` nodes maximum; inactive nodes are allowed.
- Mutation rates: node rate `0.05`, output/connection rate `0.10`.
- Connectivity: 8-connected components for object counting and matching.
- Endpoint: fixed binary endpoint using threshold `128`. Internal operations and thresholds may still evolve, but the final output must be a binary mask.
- Classification: not included.
- Fitness: `loss = 1 - mean(Q_seg)` over the training images for that run.
- Metrics: Dice, IoU, object-count error, `Q_seg`, and `R_seg`.
- Primitive set: `illumination_core_v1`, consisting of `Identity`, `Inverse`, `Normalize`, `GaussianBlur`, `MedianBlur`, `OtsuThreshold`, `Threshold`, `InRange`, `Sobel`, `Erode`, `Dilate`, `Open`, `Close`, `Fill`, `RmSmallObjects`, and `ClearBorder`.

The endpoint is fixed so every candidate produces a comparable binary segmentation. Connectivity refers to connected-component analysis, not graph connectivity.

## Required code locations

Keep the existing pilot implementation unchanged. Add or extend the final-experiment code in these locations:

```text
projects/illumination_robustness/evolution/fitness.py
projects/illumination_robustness/evolution/experiment_runner.py
projects/illumination_robustness/data/experiment_splits.py
scripts/run_scene_holdout.py
scripts/run_easy_to_hard.py
scripts/reevaluate_pipeline.py
scripts/summarize_final_results.py
```

The experiment runners are implemented as follows:

```text
projects/illumination_robustness/evolution/experiment_runner.py
scripts/run_scene_holdout.py
scripts/run_easy_to_hard.py
scripts/reevaluate_pipeline.py
scripts/summarize_final_results.py
```

The runners load Kartezio directly from the repository's `src/` directory, so a separate Kartezio checkout is not required.

The current runner implementation already performs evolution, writes the resolved configuration and split, exports `elite.json` and `elite.txt`, evaluates train and test images, writes per-image CSV files and an aggregate summary, and saves a fitness-history figure. Prediction panels, robustness-series CSV generation, and failure-case figures remain required follow-up artifact work before the final study is considered complete.

Configuration files belong in:

```text
configs/illumination_robustness/common.json
configs/illumination_robustness/scene_holdout_one.json
configs/illumination_robustness/scene_holdout_two.json
configs/illumination_robustness/easy_to_medium_hard.json
configs/illumination_robustness/easy_medium_to_hard.json
```

The common configuration must record the benchmark root, primitive-set name, node capacity, population size, generation count, mutation rates, endpoint, metric protocol, loss definition, connectivity, classification flag, and seed list. Experiment-specific files must record the scene or condition split rules.

## Dataset and identifiers

The benchmark has four scenes, thirteen shading levels, and fourteen noise levels. A condition is identified by `(scene, b, n)`, where `b` is the shading index and `n` is the noise index. Each scene therefore has 182 images. Use the repository benchmark manifest rather than constructing filenames independently.

Every generated split must save its explicit image identifiers. The split validator must confirm that training and test image IDs do not overlap and that all requested files and ground-truth masks exist.

## Experiment A: one-scene holdout

This is full-condition leave-one-scene-out evaluation. Each run trains on all 182 conditions from three scenes and tests on all 182 conditions from the remaining scene.

| Rotation | Training scenes | Test scene | Train images | Test images |
|---|---|---:|---:|---:|
| `scene1_to_scene4` | 1, 2, 3 | 4 | 546 | 182 |
| `scene2_to_scene3` | 1, 2, 4 | 3 | 546 | 182 |
| `scene3_to_scene2` | 1, 3, 4 | 2 | 546 | 182 |
| `scene4_to_scene1` | 2, 3, 4 | 1 | 546 | 182 |

Run each rotation with all three seeds: 12 runs total. Recommended IDs are `loso1_train123_test4_seed7`, `loso1_train123_test4_seed123`, and `loso1_train123_test4_seed31415`, with the analogous scene identifiers for the other rotations.

## Experiment B: two-scene holdout

This is also full-condition evaluation, but two scenes are held out together. Each run trains on 364 images from two scenes and tests on 364 images from the other two.

| Rotation | Training scenes | Test scenes |
|---|---|---|
| `train12_test34` | 1, 2 | 3, 4 |
| `train13_test24` | 1, 3 | 2, 4 |
| `train14_test23` | 1, 4 | 2, 3 |
| `train23_test14` | 2, 3 | 1, 4 |
| `train24_test13` | 2, 4 | 1, 3 |
| `train34_test12` | 3, 4 | 1, 2 |

Run every rotation with all three seeds: 18 runs total. IDs must include the scene sets and seed, for example `loso2_train12_test34_seed7`.

## Experiment C: easy-to-hard condition transfer

These runs use all four scenes in both training and testing, but condition IDs must be disjoint. They measure transfer across degradation levels rather than scene generalisation.

Use this deterministic difficulty score for every condition:

```text
difficulty(b,n) = ((b-1)/12 + (n-1)/13) / 2
```

Assign every one of the 182 conditions per scene as follows:

- Easy: `difficulty <= 1/3` — 45 conditions per scene.
- Medium: `1/3 < difficulty <= 2/3` — 93 conditions per scene.
- Hard: `difficulty > 2/3` — 44 conditions per scene.

This partition uses all 182 conditions per scene, with no omissions or overlaps. Across four scenes, the groups contain 180 Easy, 372 Medium, and 176 Hard images. The full benchmark grid remains available for scene-holdout experiments, while these groups provide a complete deterministic condition-transfer partition.

### C1: Easy to medium and hard

Train once on Easy and test the same saved elite pipeline on the combined Medium+Hard set. This uses 180 training images and 548 test images. Report the combined test result and separate Medium and Hard summaries from the per-image results. Use IDs `easy_to_medium_hard_seed7`, `easy_to_medium_hard_seed123`, and `easy_to_medium_hard_seed31415`. This is 3 runs total.

### C2: Easy plus medium to hard

Train on Easy plus Medium, then test on Hard. This uses 552 training images and 176 test images. Use IDs `easy_medium_to_hard_seed7`, `easy_medium_to_hard_seed123`, and `easy_medium_to_hard_seed31415`. This is 3 runs total.

The complete planned set is therefore 36 runs: 12 one-scene holdout, 18 two-scene holdout, 3 Easy-to-Medium+Hard, and 3 Easy+Medium-to-Hard.

## Robustness reporting

`Q_seg` is calculated for every image. For the prescribed adjacent-condition series, report:

```text
R_seg(scene) = (1/13) * sum over b=1..13 of Q_seg(scene, b, b+1)
```

`R_seg` is a reporting metric, not a second fitness strategy. For scene-holdout runs, calculate it on the held-out scene(s). For condition-transfer runs, report per-group metrics and calculate `R_seg` only when the required adjacent conditions are present in the evaluated subset; do not silently fill missing conditions.

## Result directory and naming convention

Every run must write to its own directory:

```text
results/final/<experiment>/<run_id>/
  config.json
  split.json
  metadata.json
  command.txt
  log.txt
  pipeline/
    elite.json
    elite.txt
  metrics/
    train_per_image.csv
    test_per_image.csv
    full_grid.csv
    robustness_series.csv
    scene_summary.csv
    aggregate_summary.json
  figures/
    fitness_history.png
    predictions.png
    robustness_curve.png
    failure_cases/
```

Use these experiment directory names:

```text
results/final/scene_holdout_one/
results/final/scene_holdout_two/
results/final/easy_to_hard/
```

For the current one-scene Easy-to-Medium+Hard rotations, use this canonical location:

```text
results/final/easy_to_hard_scene_rotations/
  easy_to_medium_hard_train1_seed7/
  easy_to_medium_hard_train2_seed7/
  easy_to_medium_hard_train3_seed7/
  easy_to_medium_hard_train4_seed7/
```

The run-name convention is:

```text
<experiment>_train<scene>_seed<seed>/
```

For example, `easy_to_medium_hard_train1_seed7` means that Easy images from scene 1 were used for evolution, while the combined Medium+Hard test set was evaluated across scenes 1--4. Additional seeds use the same directory with only the seed suffix changed. Do not create nested output roots with the same experiment name.

The run directory is the authoritative record. `config.json` is the resolved configuration, not only the template. `split.json` must contain the experiment name, run ID, seed, training and test scenes, condition rules, explicit training and test image IDs, and validation results. `metadata.json` must include the commit identifier, timestamp, Python/package versions, primitive-set version, node capacity, population, generations, mutation rates, endpoint, metric protocol, and completion status.

## Required per-image fields

Each metric CSV must include `run_id`, experiment name, seed, scene, shading index, noise index, image ID, split, condition group, Dice, IoU, `Q_seg`, object-count error, ground-truth object count, predicted object count, and status. Do not report only aggregate means; retain raw rows so summaries can be recomputed.

## Aggregation and visualisation

After all runs, write combined outputs to:

```text
results/final/aggregate/
  scene_holdout_one_summary.csv
  scene_holdout_two_summary.csv
  easy_to_medium_summary.csv
  easy_to_hard_summary.csv
  easy_medium_to_hard_summary.csv
  seed_summary.csv
  final_comparison.csv
```

Aggregate by run, seed, scene, and experiment. Report mean and standard deviation across the three seeds. Save prediction visualisations for test images, fitness histories, robustness curves where valid, and representative failure cases including strong shading, high noise, combined degradation, touching objects, border objects, fragmented masks, and empty masks.

## Reproducibility requirements

Before each run, save the resolved configuration and split, seed NumPy before creating the Kartezio trainer, and record the exact command. The trainer does not require a separate seed argument; the NumPy seed controls the initial population and stochastic mutations/connections. Do not select a pipeline using test images. The elite pipeline must be exported before test evaluation, and `reevaluate_pipeline.py` must be able to reproduce the saved metrics from `elite.json` and `split.json`.

## Execution order

1. Validate manifests, ground truth, primitive registration, endpoint, and split generation.
2. Run one smoke rotation for each experiment type with seed `7`.
3. Confirm that train/test IDs are disjoint and that all expected image counts are present.
4. Run all 39 primary jobs with seeds `7`, `123`, and `31415`.
5. Re-evaluate every saved elite from its saved split.
6. Generate aggregate CSV/JSON files and figures.
7. Inspect failed or incomplete runs before comparing experiments.

No result is considered complete unless its configuration, split, command, log, elite pipeline, raw per-image metrics, and required figures are present.

## How to run the experiments

Run all commands from the repository root. Python 3.11 or newer is required. Install the repository and all declared dependencies with:

```powershell
python -m pip install -e .
```

The declared dependencies include NumPy, pandas, SciPy, tifffile, scikit-image, OpenCV, numba, matplotlib, czifile, tabulate, roifile, and codecarbon. SciPy is required for the MATLAB ground-truth files, and tifffile is required for the benchmark images. MATLAB is not required for the Python evolution runs.

The configuration files are:

```text
configs/illumination_robustness/common.json
configs/illumination_robustness/scene_holdout_one.json
configs/illumination_robustness/scene_holdout_two.json
configs/illumination_robustness/easy_to_medium_hard.json
configs/illumination_robustness/easy_medium_to_hard.json
```

The scripts accept configuration paths, output roots, optional rotation/run selectors, optional seed selectors, and optional generation overrides.

### Smoke runs

Run one short validation job before launching the full set. The smoke mode must use the same primitive set, split generation, metrics, artifact layout, and endpoint as the full run, but may use a reduced generation count only for validation.

```powershell
python scripts/run_scene_holdout.py `
  --config configs/illumination_robustness/scene_holdout_one.json `
  --common-config configs/illumination_robustness/common.json `
  --rotation train123_test4 `
  --seed 7 `
  --generations 5 `
  --output-root results/final/scene_holdout_one `
  --smoke

python scripts/run_scene_holdout.py `
  --config configs/illumination_robustness/scene_holdout_two.json `
  --common-config configs/illumination_robustness/common.json `
  --rotation train12_test34 `
  --seed 7 `
  --generations 5 `
  --output-root results/final/scene_holdout_two `
  --smoke

python scripts/run_easy_to_hard.py `
  --config configs/illumination_robustness/easy_to_medium_hard.json `
  --common-config configs/illumination_robustness/common.json `
  --run easy_to_medium `
  --seed 7 `
  --generations 5 `
  --output-root results/final/easy_to_medium_hard `
  --smoke
```

The smoke run is valid only when it creates a complete result directory, validates disjoint train/test IDs, exports an elite pipeline, evaluates the test split, and writes the required metrics and figures. Do not use smoke results in the final comparison. The smoke commands use five generations while preserving the population size, primitive library, endpoint, split logic, metrics, and artifact layout of a full run.

### Full one-scene holdout

Run all four rotations for each of the three seeds with 500 generations:

```powershell
python scripts/run_scene_holdout.py `
  --config configs/illumination_robustness/scene_holdout_one.json `
  --common-config configs/illumination_robustness/common.json `
  --output-root results/final/scene_holdout_one
```

This creates 12 runs under `results/final/scene_holdout_one/`.

### Full two-scene holdout

Run all six rotations for each of the three seeds:

```powershell
python scripts/run_scene_holdout.py `
  --config configs/illumination_robustness/scene_holdout_two.json `
  --common-config configs/illumination_robustness/common.json `
  --output-root results/final/scene_holdout_two
```

This creates 18 runs under `results/final/scene_holdout_two/`.

### Full Easy-to-Medium+Hard run

Run both condition-transfer experiments for all three seeds:

```powershell
python scripts/run_easy_to_hard.py `
  --config configs/illumination_robustness/easy_to_medium_hard.json `
  --common-config configs/illumination_robustness/common.json `
  --output-root results/final/easy_to_hard
```

This creates three runs: each is trained on Easy and tested on the combined Medium+Hard set.

### Full Easy+Medium-to-Hard runs

```powershell
python scripts/run_easy_to_hard.py `
  --config configs/illumination_robustness/easy_medium_to_hard.json `
  --common-config configs/illumination_robustness/common.json `
  --output-root results/final/easy_to_hard
```

This creates three additional runs.

To run one selected rotation or seed, pass its explicit selector. For example:

```powershell
python scripts/run_scene_holdout.py `
  --config configs/illumination_robustness/scene_holdout_one.json `
  --common-config configs/illumination_robustness/common.json `
  --rotation train123_test4 `
  --seed 123 `
  --output-root results/final/scene_holdout_one
```

For a smoke run of the Easy+Medium-to-Hard experiment:

```powershell
python scripts/run_easy_to_hard.py `
  --config configs/illumination_robustness/easy_medium_to_hard.json `
  --common-config configs/illumination_robustness/common.json `
  --run easy_medium_to_hard `
  --seed 7 `
  --generations 5 `
  --output-root results/final/easy_to_hard `
  --smoke
```

If a single rotation or seed needs to be repeated, pass its explicit selector. Do not overwrite a completed run silently; use a new recovery directory or remove the old run only after confirming that it is incomplete.

## Elite pipeline saving requirements

The runner must save the final selected individual immediately after evolution and before test evaluation. The saved pipeline is the only pipeline used for the test split. Each run must contain:

```text
pipeline/elite.json
pipeline/elite.txt
```

`elite.json` must contain the complete serialisable graph: primitive names, node indices, parameters, input connections, active-node information, endpoint configuration, primitive-set version, and fitness value. `elite.txt` must be a human-readable graph summary showing the active operations in execution order. If the framework provides a native pipeline serializer, use it; otherwise save an equivalent explicit representation that can be reconstructed without the original in-memory trainer.

The runner must also save:

- `pipeline/elite_train_predictions/` when train predictions are needed for audit.
- `figures/predictions.png` containing test predictions and ground truth.
- `figures/fitness_history.png` containing generation and best-fitness values.
- `figures/robustness_curve.png` when the evaluated split contains the required robustness sequence.
- `figures/failure_cases/` with the selected difficult examples and their masks.

Test predictions must be generated from the selected elite before the run is marked complete. The saved `pipeline/elite.json` is the authoritative machine-readable artifact, and `pipeline/elite.txt` is the human-readable description. The re-evaluation script validates that these files and the saved split and metric files are present.

## Required command and log capture

Every run must write the exact command line to `command.txt` and redirect normal progress and errors to `log.txt`. Progress should identify the run ID, seed, generation, best loss, and completed image count. A failed run must retain its partial log and `metadata.json` status rather than being presented as complete.

After all jobs finish, generate the aggregate reports:

```powershell
python scripts/summarize_final_results.py `
  --input-root results/final `
  --output-dir results/final/aggregate
```

Before accepting the results, re-evaluate all saved elites:

```powershell
python scripts/reevaluate_pipeline.py `
  --input-root results/final `
  --check-original-metrics
```

The aggregate step must clearly report incomplete runs or missing artifacts. A final comparison should be made only from the saved per-image CSV files and the re-evaluation outputs. `reevaluate_pipeline.py --run <run_id>` reconstructs the saved elite and regenerates exact full-grid predictions and metrics; without `--run`, the utility performs artifact validation.

## Current scene-rotation run protocol

The current feasibility runs use the selectable hybrid fitness objective:

```text
L_hybrid = 0.8(1 - Dice) + 0.2(object-count error)
```

The shared configuration remains population `16`, but the current scene-rotation comparison overrides this to population `32` at the command line. The current comparison settings are:

```text
Nodes: 8
Population: 32
Generations: 50
Seed: 7
Mutation node rate: 0.20
Mutation output rate: 0.20
Fitness: hybrid_dice_count
Training: Easy images from one selected scene
Testing: Medium+Hard images from scenes 1--4
```

Run all four rotations with:

```powershell
foreach ($scene in 1,2,3,4) {
  python scripts/run_easy_to_hard.py `
    --config configs/illumination_robustness/easy_to_medium_hard.json `
    --common-config configs/illumination_robustness/common.json `
    --run easy_to_medium_hard `
    --train-scenes $scene `
    --test-scenes 1,2,3,4 `
    --seed 7 `
    --generations 50 `
    --population-size 32 `
    --fitness-objective hybrid_dice_count `
    --mutation-node-rate 0.20 `
    --mutation-output-rate 0.20 `
    --output-root results/final/easy_to_hard_scene_rotations_hybrid_v2
}
```

The corresponding result layout is:

```text
results/final/easy_to_hard_scene_rotations_hybrid_v2/
  easy_to_medium_hard_train1_seed7/
  easy_to_medium_hard_train2_seed7/
  easy_to_medium_hard_train3_seed7/
  easy_to_medium_hard_train4_seed7/
```

The earlier completed rotation results are stored in:

```text
results/final/easy_to_hard_scene_rotations/
```

## Full-grid evaluation of saved elites

After evolution, each saved elite can be evaluated on all 728 benchmark images. This evaluation does not change the elite or its fitness. It writes `metrics/full_grid.csv` and `metrics/full_grid_summary.json` inside the selected run directory.

```powershell
$root = "results/final/easy_to_hard_scene_rotations"
foreach ($run in `
  "easy_to_medium_hard_train1_seed7",
  "easy_to_medium_hard_train2_seed7",
  "easy_to_medium_hard_train3_seed7",
  "easy_to_medium_hard_train4_seed7"
) {
  python scripts/reevaluate_pipeline.py `
    --input-root $root `
    --run $run
}
```

Full-grid outputs are stored at:

```text
results/final/easy_to_hard_scene_rotations/<run_id>/metrics/full_grid.csv
results/final/easy_to_hard_scene_rotations/<run_id>/metrics/full_grid_summary.json
```

Full-grid scores must be reported separately from the held-out Medium+Hard scores because they include conditions related to the evolution training split. They are used for complete benchmark coverage and comparison with the 728-image baseline results.

## Shading-stratified 45-image-per-scene experiment

Configuration:

```text
configs/illumination_robustness/shading_stratified_45.json
```

Each scene contributes 45 training conditions:

```text
15 lower-shading:  b = 1..5,  n = 1..3
15 medium-shading: b = 5..9,  n = 6..8
15 higher-shading: b = 9..13, n = 12..14
```

All four scenes contribute 180 training images. Run it with:

```powershell
python scripts/run_easy_to_hard.py `
  --config configs/illumination_robustness/shading_stratified_45.json `
  --common-config configs/illumination_robustness/common.json `
  --run shading_stratified_45 `
  --seed 7 `
  --generations 50 `
  --population-size 32 `
  --fitness-objective hybrid_dice_count `
  --mutation-node-rate 0.20 `
  --mutation-output-rate 0.20 `
  --output-root results/final/shading_stratified_45
```

Results are stored in `results/final/shading_stratified_45/shading_stratified_45_seed7/`. The full-grid test output is post-evolution evaluation only and must not influence fitness or elite selection.

## Shading and noise sensitivity analysis

After full-grid evaluation, analyze each elite by shading level and noise level:

```powershell
python scripts/analyze_condition_sensitivity.py `
  --results-root results/final/easy_to_hard_scene_rotations
```

The analysis reads each run's `metrics/full_grid.csv` and writes:

```text
results/final/easy_to_hard_scene_rotations/sensitivity_analysis/
  by_shading.csv
  by_noise.csv
  by_scene_shading.csv
  by_scene_noise.csv
  qseg_by_shading.png
  qseg_by_noise.png
  summary.json
```

The tables report mean and standard deviation for Dice, IoU, `Q_seg`, and count error. The scene-specific tables allow sensitivity to be separated from scene difficulty. A sharper decline across shading levels indicates illumination sensitivity; a sharper decline across noise levels indicates noise sensitivity. Use the same image coverage for every pipeline when comparing these curves.
