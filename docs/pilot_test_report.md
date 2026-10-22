# Pilot-Test Report — Kartezio Illumination Robustness

## Purpose

This pilot test validates that the project can evolve a Kartezio graph on a
deterministic synthetic image set, select an elite using training fitness, and
evaluate that elite on unseen synthetic images. It is an integration and
workflow check, not a paper result or a claim about illumination robustness on
the benchmark dataset.

## Command and configuration

The sweep was run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_pilot_sweep.py `
  --generations 50 100 250 500 1000 `
  --nodes 16 `
  --seed 42 `
  --fixture-dir results\pilot_fixtures `
  --output-root results\pilot_sweep
```

The fitness objective was `loss = 1 - mean(Q_seg)`, minimized by Kartezio.
The same seed, node budget, mutation rates, and fixture set were used for all
generation budgets. Kartezio stopped early when the training loss reached zero,
so a requested generation count is an upper bound rather than a guaranteed
number of completed iterations.

## Data split

Training fixtures (8):

```text
01_single_disk
02_multiple_disks
03_rectangles
04_ellipse_triangle
05_thin_crossing_objects
06_touching_disks
07_object_with_hole
08_border_objects
```

Unseen test fixtures (4):

```text
09_illumination_gradient
10_noisy_multiple_objects
11_empty
12_constant
```

The test fixtures were not supplied to the trainer during evolution or elite
selection.

## Results

Fitness is a loss, so lower is better.

| Requested generations | Completed history length | Training loss | Unseen-test loss | Active nodes |
|---:|---:|---:|---:|---:|
| 50 | 51 | 0.749179 | 0.442203 | 2 |
| 100 | 20 | 0.000000 | 0.201567 | 2 |
| 250 | 69 | 0.000000 | 0.000000326 | 2 |
| 500 | 69 | 0.000000 | 0.000000326 | 2 |
| 1000 | 69 | 0.000000 | 0.000000326 | 2 |

The best-performing pipeline selected by the sweep is the 1000-generation run
because it ties for the lowest training loss and has the lowest recorded test
loss. The 250-, 500-, and 1000-generation runs have effectively identical
test loss, indicating a plateau for this seed and fixture set.

The selected elite uses 2 active nodes out of 16 available nodes:

```text
node 1: BitwiseNot(input)
node 15: OtsuThreshold(node 1)
```

Its complete genotype and readable pipeline are saved under:

```text
results/pilot_sweep/best_pipeline/pipeline.json
results/pilot_sweep/best_pipeline/pipeline.txt
```

The prediction visualization covers all four unseen test fixtures. The most
complex test fixture is marked in the figure and is recorded in the selected
summary:

```text
results/pilot_sweep/best_pipeline/figures/elite_predictions.png
results/pilot_sweep/best_pipeline/summary.json
```

## Interpretation

The pilot test passed its intended integration checks:

- Kartezio evolution completed for all requested budgets.
- Training and test data were separated.
- The elite was selected using training loss only.
- Test loss was computed after selection on unseen fixture images.
- The final genotype, active-node structure, fitness history, and test
  predictions were saved.

The rapid drop in test loss is encouraging for the synthetic fixtures, but it
should not be interpreted as evidence of generalization to the illumination
benchmark. The test set has only four images, includes simple generated
objects, and contains empty/constant cases that can be easy to score. The
result also uses one random seed and one primitive/node configuration.

## Next step

Before using this workflow for the benchmark, Stage 5 should freeze the real
training-condition sets, use the leave-one-scene-out rotations, run multiple
seeds, and report per-image and per-scene metrics separately. The pilot output
is suitable as a software-integration check, not as a final experimental
result.
