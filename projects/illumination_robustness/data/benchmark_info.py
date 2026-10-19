"""Fixed facts about the Khan et al. (2015) hardware-items benchmark.

Every constant carries its source. Keys: [P] main paper, [S1] S1 Text (Table A).
See ``docs/implementation_decisions.md``.
"""

from __future__ import annotations

from typing import Final

# [P, Materials and Methods]: R = 4 data sets, B = 13 shading grades, N = 14 noise grades.
N_SCENES: Final[int] = 4
N_SHADING: Final[int] = 13
N_NOISE: Final[int] = 14
SCENES: Final[tuple[int, ...]] = tuple(range(1, N_SCENES + 1))
SHADING_LEVELS: Final[tuple[int, ...]] = tuple(range(1, N_SHADING + 1))
NOISE_LEVELS: Final[tuple[int, ...]] = tuple(range(1, N_NOISE + 1))

# [P]: m = 1000 rows, n = 1500 columns, 8-bit grey values.
IMAGE_SHAPE: Final[tuple[int, int]] = (1000, 1500)

# Relative locations inside the benchmark root (``hardware_items/``) as distributed.
DATA_SUBDIR: Final[str] = "data"
TRUTH_SUBDIR: Final[str] = "truth"

# Scene-level annotation files and the MATLAB variable names they were found to
# contain on inspection (2026-10-19). Names are checked, never guessed, at load time.
GT_MASK_FILENAME: Final[str] = "benchdata_{scene}_gd.mat"
GT_MASK_VARIABLE: Final[str] = "ground_truth"
GT_CLASS_FILENAME: Final[str] = "benchdata_{scene}_class.mat"
GT_CLASS_VARIABLE: Final[str] = "im_label"

# [S1, Table A]: number of objects per class k = 1..9 for each scene. Used only to
# audit ground-truth BLOB counts; the metric uses BLOBs from the mask itself (D7).
TABLE_A_OBJECTS_PER_CLASS: Final[dict[int, tuple[int, ...]]] = {
    1: (8, 4, 3, 5, 7, 0, 0, 0, 0),
    2: (10, 5, 5, 5, 4, 3, 0, 0, 0),
    3: (21, 8, 8, 9, 4, 2, 0, 0, 0),
    4: (20, 11, 6, 8, 6, 2, 3, 1, 1),
}
TABLE_A_TOTAL_OBJECTS: Final[dict[int, int]] = {
    scene: sum(counts) for scene, counts in TABLE_A_OBJECTS_PER_CLASS.items()
}

# [P, Eq. 3]: robustness image series X(r, b, b+1), b = 1..13 (single images, D1).
ROBUSTNESS_SERIES: Final[tuple[tuple[int, int], ...]] = tuple(
    (b, b + 1) for b in range(1, N_SHADING + 1)
)


def image_id(scene: int, shading: int, noise: int) -> str:
    """Return the stable project image identifier, e.g. ``r01_b02_n03``.

    Args:
        scene: Scene index r in 1..4.
        shading: Shading index b in 1..13.
        noise: Noise index n in 1..14.

    Returns:
        Identifier string used in manifests, splits and result tables.
    """
    return f"r{scene:02d}_b{shading:02d}_n{noise:02d}"


def benchmark_filename(scene: int, shading: int, noise: int) -> str:
    """Return the distributed image filename, e.g. ``benchdata_01_02_03.tif`` [P, p.3]."""
    return f"benchdata_{scene:02d}_{shading:02d}_{noise:02d}.tif"
