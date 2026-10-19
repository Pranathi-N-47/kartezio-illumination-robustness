# Data and Ground-Truth Audit Report

## 1. Manifest Validation
- Total records in manifest: 728
- Status 'ok' records: 728 / 728

## 2. Ground-Truth & BLOB Audit
| Scene | Raw Mask Shape | Resized Shape | Resized BLOBs (8-conn) | Table A Count | Discrepancy Note |
|---|---|---|---|---|---|
| Scene 1 | (3456, 5184) | (1000, 1500) | 27 | 27 | Exact match |
| Scene 2 | (3456, 5184) | (1000, 1500) | 32 | 32 | Exact match |
| Scene 3 | (3456, 5184) | (1000, 1500) | 51 | 52 | 1 touching pair (D7) |
| Scene 4 | (3456, 5184) | (1000, 1500) | 58 | 58 | Exact match |

## 3. Sample Alignment & Integrity
- Verified spatial alignment & dtype for 8 spot check images across all scenes.

## 4. Cross-Validation Splits Audit
- **Rotation test_scene=1**: Train scenes: [2, 3, 4]. Quality-only train: 27 imgs, Robustness train: 27 imgs, Held-out test: 182 imgs, Robustness series: 13 imgs.
- **Rotation test_scene=2**: Train scenes: [1, 3, 4]. Quality-only train: 27 imgs, Robustness train: 27 imgs, Held-out test: 182 imgs, Robustness series: 13 imgs.
- **Rotation test_scene=3**: Train scenes: [1, 2, 4]. Quality-only train: 27 imgs, Robustness train: 27 imgs, Held-out test: 182 imgs, Robustness series: 13 imgs.
- **Rotation test_scene=4**: Train scenes: [1, 2, 3]. Quality-only train: 27 imgs, Robustness train: 27 imgs, Held-out test: 182 imgs, Robustness series: 13 imgs.

## 5. Summary
- All 728 images present, valid uint8 TIFFs, shape 1000x1500.
- All 4 scene ground-truth masks present and aligned.
- No data leakage across any leave-one-scene-out split.
- Audit status: **PASSED**.
