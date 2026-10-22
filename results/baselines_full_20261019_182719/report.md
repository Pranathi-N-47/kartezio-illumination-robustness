# Baseline Evaluation Report — baselines_full_20261019_182719

## Executive Summary
- **Mode:** `full` (728 images per method)
- **Execution Time:** 511.87 seconds
- **Git Commit:** `93c7a151a5e6d0b404b8ed85ece2884eb14734c4`

## Scene-Level Results Table

| Method | Scene | Images | Mean Dice | Mean IoU | Mean Q_seg | R_seg (13-img) | Mean Runtime (ms) |
|---|---|---|---|---|---|---|---|
| **OTSU** | Scene 1 | 182 | 0.5598 ± 0.3392 | 0.4637 ± 0.3213 | 0.4566 | 0.3919 | 286.9 |
| **OTSU** | Scene 2 | 182 | 0.4725 ± 0.3227 | 0.3682 ± 0.2799 | 0.3627 | 0.3039 | 283.8 |
| **OTSU** | Scene 3 | 182 | 0.5451 ± 0.2908 | 0.4254 ± 0.2556 | 0.2533 | 0.2268 | 285.3 |
| **OTSU** | Scene 4 | 182 | 0.6206 ± 0.2225 | 0.4810 ± 0.1944 | 0.3378 | 0.2619 | 310.7 |
| **SOBEL** | Scene 1 | 182 | 0.5195 ± 0.3967 | 0.4495 ± 0.3710 | 0.4518 | 0.4092 | 263.5 |
| **SOBEL** | Scene 2 | 182 | 0.4969 ± 0.3768 | 0.4131 ± 0.3310 | 0.4349 | 0.3866 | 259.2 |
| **SOBEL** | Scene 3 | 182 | 0.4799 ± 0.3681 | 0.3925 ± 0.3187 | 0.3063 | 0.2821 | 260.3 |
| **SOBEL** | Scene 4 | 182 | 0.5436 ± 0.3332 | 0.4377 ± 0.2829 | 0.3492 | 0.3357 | 289.5 |

## Robustness Curves & Degradation
![Robustness Curve](figures/quality_vs_artifact_level.png)

## Visual Comparison of Segmentations
![Visual Comparison](figures/baseline_segmentation_comparison.png)

## Artifacts Created
- CSV: `metrics/per_image.csv` (1456 rows)
- Summary: `metrics/per_scene.csv`
- Config: `config.json`
- Metadata: `metadata.json`
