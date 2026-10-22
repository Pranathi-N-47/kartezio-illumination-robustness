# MATLAB baseline reference

`baselines.m` contains the segmentation-only MATLAB-compatible Otsu and Sobel
reference implementation. It deliberately excludes classification quality
(`mu3`) and writes `Q_seg = mu1 * mu2` together with Dice and IoU.

The Python implementation that preceded this reference is preserved at
`extras/baselines_python.py` and is not the authoritative paper baseline.
