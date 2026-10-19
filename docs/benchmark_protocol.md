# Benchmark Protocol — Segmentation-Only Illumination Robustness

**Status:** FROZEN (Stage 2, 2026-10-19). Any future change must be
recorded as a deviation in `docs/implementation_decisions.md`.

Source keys ([P], [S1]–[S4], [M]) and decision IDs (D1…) refer to
`docs/implementation_decisions.md`.

## 1. Data

- Images: `hardware_items/data/benchdata_RR_BB_NN.tif`, r ∈ 1..4 (scene),
  b ∈ 1..13 (shading; 1 = bright, 13 = dark/heavy shading), n ∈ 1..14 (noise;
  Gaussian σ_n² = 125·(n − 1)) [P, Eq. 1–2]. uint8, 1000×1500.
- Stable image ID: `r{RR}_b{BB}_n{NN}` (e.g. `r01_b02_n03`).
- Ground truth: one binary mask per scene, `hardware_items/truth/benchdata_{r}_gd.mat`
  (`ground_truth`), applied to all conditions of that scene.

## 2. Ground-truth preprocessing (D8)

1. Nonzero → foreground (explicit `to_binary_foreground`).
2. Resize 3456×5184 → 1000×1500 with area averaging; foreground = value ≥ 0.5.
3. No morphological opening.
4. GT BLOBs T = 8-connected components of the resized mask (D7).

## 3. Image normalisation (D9)

`x' = 1 − (clip(x, p2, p98) − p2) / (p98 − p2)`, with p2/p98 the Hazen percentiles of
the image. Output float64 in [0, 1], objects bright. If `p98 == p2` (constant or
near-constant image), output is all zeros. NaN/Inf inputs raise `ValueError`.

## 4. Foreground convention

Predictions and GT are binary; any nonzero prediction pixel is foreground.

## 5. Segmentation-only quality measures [S3, S4]

Fuzzy membership [S3, Eq. 2], s = 2:

```text
µ(θ; s, α, β) = 1                                   θ < α
              = 1 − 2^(s−1) ((θ − α)/(β − α))^s     α ≤ θ ≤ (α+β)/2
              = 2^(s−1) ((β − θ)/(β − α))^s         (α+β)/2 < θ ≤ β
              = 0                                   θ > β
```

**Segmentation measure 1** [S4, Eqs. 1–3] with matching rule D6:

```text
q1 = |card(T) − card(S_c)| / card(T)
µ1 = µ(q1; 2, 0, 1)                (D3: no leading "1 −")
```

**Segmentation measure 2** [S4, Eqs. 4–5] (D5):

```text
q2 = |G XOR P| / |G ∪ P|  = 1 − IoU
µ2 = µ(q2; 2, 0, 1)
```

**Segmentation quality** [S4, Eq. 9]: `Q_seg(r, b, n) = µ1 · µ2`.

**Segmentation robustness** (adapted from [P, Eq. 3], D1):

```text
R_seg(r) = (1/13) Σ_{b=1..13} Q_seg(r, b, b+1)
```

`R_seg` is segmentation-only. It is **not** comparable to the paper's full R
(R_otsu = 0.54, R_edge = 0.47 for r = 1 include classification).

**Artifact level** [S3, Eqs. 1–4]:

```text
µ_shading = µ(x_m; 2, 0, 255)          x_m = mean pixel value of the raw image
µ_noise   = 1 − µ(σ_b; 2, 10, 45)      σ_b = sqrt(125 · (n − 1))
A(r,b,n)  = 1 − (1 − µ_shading)(1 − µ_noise)
```

## 6. Supplementary verified metrics

```text
Dice = 2TP / (2TP + FP + FN)
IoU  = TP / (TP + FP + FN)
```

Empty-mask policy (all overlap metrics, including q2):
- both masks empty → Dice = IoU = 1, q2 = 0;
- exactly one empty → Dice = IoU = 0, q2 = 1.

## 7. Evaluation splits

Leave-one-scene-out (4 rotations). The robustness series
`{(b, b+1) : b = 1..13}` and the full 182-condition grid of the held-out scene are
reported separately. Training-condition subsets are defined in Stage 5 from condition
indices only (never from results).
