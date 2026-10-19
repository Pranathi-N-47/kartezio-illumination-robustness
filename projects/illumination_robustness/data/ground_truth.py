"""Ground-truth loading, normalisation of representation and alignment checks.

Ground truth in this benchmark is per scene: a binary mask at camera resolution
(3456×5184) and a class map. The original arrays are kept untouched in
:class:`RawSceneAnnotation`; every transformation happens in an explicit function
whose behaviour is documented in ``docs/benchmark_protocol.md`` §2 and decision D8.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.io

from projects.illumination_robustness.data import benchmark_info as info
from projects.illumination_robustness.data.manifest import scene_annotation_paths
from projects.illumination_robustness.evaluation.object_metrics import label_blobs

# Two axes are considered to share one scale factor if they differ by less than this
# relative tolerance. The benchmark's factor is exactly 3.456 on both axes; any real
# mismatch would indicate a cropped or differently framed annotation.
_SCALE_RELATIVE_TOLERANCE = 0.01


@dataclass(frozen=True)
class GroundTruthConfig:
    """Explicit ground-truth preprocessing policy (D8).

    Attributes:
        target_shape: ``(rows, cols)`` of the benchmark images.
        resize_method: Only ``"area"`` (OpenCV ``INTER_AREA``) is supported.
        foreground_threshold: Resized mask values ``>=`` this are foreground.
        connectivity: 4 or 8, used to define ground-truth BLOBs.
    """

    target_shape: tuple[int, int] = info.IMAGE_SHAPE
    resize_method: str = "area"
    foreground_threshold: float = 0.5
    connectivity: int = 8

    def __post_init__(self) -> None:
        if self.resize_method != "area":
            raise ValueError(f"Unsupported resize_method {self.resize_method!r}; only 'area'")
        if not 0.0 < self.foreground_threshold <= 1.0:
            raise ValueError("foreground_threshold must be in (0, 1]")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")


@dataclass(frozen=True)
class RawSceneAnnotation:
    """Annotation arrays exactly as stored in the ``.mat`` files.

    Attributes:
        scene: Scene index r.
        mask: Original binary mask array (not modified).
        class_map: Original class map, or ``None`` if the file is absent.
        mask_path: Source of ``mask``.
        class_map_path: Source of ``class_map``, or ``None``.
    """

    scene: int
    mask: np.ndarray
    class_map: np.ndarray | None
    mask_path: Path
    class_map_path: Path | None


@dataclass(frozen=True)
class SceneGroundTruth:
    """Ground truth aligned to image resolution, ready for evaluation.

    Attributes:
        scene: Scene index r.
        foreground: Boolean mask, shape ``config.target_shape``.
        blob_labels: int32 BLOB labels of ``foreground`` (column-major order).
        n_blobs: Number of ground-truth BLOBs, V = card(T) in [S4].
        class_map: uint8 class map at image resolution (audit only) or ``None``.
        source_mask_shape: Shape of the original mask before resizing.
        config: Policy used to produce this object.
    """

    scene: int
    foreground: np.ndarray
    blob_labels: np.ndarray
    n_blobs: int
    class_map: np.ndarray | None
    source_mask_shape: tuple[int, int]
    config: GroundTruthConfig


def load_mat_variable(path: Path, variable: str) -> np.ndarray:
    """Load one named variable from a MATLAB ``.mat`` file.

    Args:
        path: ``.mat`` file path.
        variable: Exact variable name; never guessed.

    Returns:
        The variable as a NumPy array.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If ``variable`` is absent (message lists the available keys).
        ValueError: If the file is a MATLAB v7.3 (HDF5) file, which scipy cannot read.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    try:
        contents = scipy.io.loadmat(path)
    except NotImplementedError as error:
        raise ValueError(f"{path} appears to be MATLAB v7.3/HDF5; not supported") from error
    available = sorted(key for key in contents if not key.startswith("__"))
    if variable not in contents:
        raise KeyError(f"Variable {variable!r} not in {path}; available: {available}")
    return np.asarray(contents[variable])


def load_scene_annotation(benchmark_root: Path, scene: int) -> RawSceneAnnotation:
    """Load a scene's original mask and (if present) class map.

    Args:
        benchmark_root: Directory containing ``truth/`` (e.g. ``hardware_items``).
        scene: Scene index r.

    Raises:
        FileNotFoundError: If the binary mask file is missing.
        KeyError: If the expected MATLAB variable is missing.
    """
    mask_path, class_path = scene_annotation_paths(benchmark_root, scene)
    mask = load_mat_variable(mask_path, info.GT_MASK_VARIABLE)
    class_map = None
    class_map_path = None
    if class_path.is_file():
        class_map = load_mat_variable(class_path, info.GT_CLASS_VARIABLE)
        class_map_path = class_path
    return RawSceneAnnotation(scene, mask, class_map, mask_path, class_map_path)


def to_binary_foreground(annotation: np.ndarray) -> np.ndarray:
    """Convert an annotation array to a boolean foreground mask (nonzero → True).

    Args:
        annotation: 2-D numeric array (binary mask, class map or instance map).

    Returns:
        Boolean array of the same shape.

    Raises:
        ValueError: If the array is not 2-D or contains NaN/Inf.
    """
    array = np.asarray(annotation)
    if array.ndim != 2:
        raise ValueError(f"annotation must be 2-D, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("annotation contains NaN or Inf")
    return array != 0


def _check_uniform_scale(source_shape: tuple[int, ...], target_shape: tuple[int, int]) -> None:
    row_scale = source_shape[0] / target_shape[0]
    col_scale = source_shape[1] / target_shape[1]
    if abs(row_scale - col_scale) > _SCALE_RELATIVE_TOLERANCE * max(row_scale, col_scale):
        raise ValueError(
            f"Cannot align annotation {tuple(source_shape)} to image {target_shape}: "
            f"row scale {row_scale:.4f} != column scale {col_scale:.4f}"
        )


def resize_binary_mask(
    mask: np.ndarray, target_shape: tuple[int, int], foreground_threshold: float = 0.5
) -> np.ndarray:
    """Resize a boolean mask with area averaging and re-binarise.

    Area averaging gives each output pixel the foreground fraction of the source
    pixels it covers; thresholding at 0.5 assigns it to the majority class. This is
    a project-defined approximation of MATLAB ``imresize`` + ``> 0`` on uint8 (D8),
    not a bit-exact reproduction.

    Args:
        mask: 2-D boolean (or 0/1) array.
        target_shape: ``(rows, cols)``.
        foreground_threshold: Output pixels with fraction ``>=`` this are foreground.

    Returns:
        Boolean array of shape ``target_shape``.

    Raises:
        ValueError: If the mask is not 2-D or the two axes would need different scales.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if tuple(mask.shape) == tuple(target_shape):
        return mask.astype(bool, copy=True)
    _check_uniform_scale(mask.shape, target_shape)
    rows, cols = target_shape
    fraction = cv2.resize(mask.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
    return fraction >= foreground_threshold


def resize_label_map(labels: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize an integer label map with exact nearest-neighbour sampling.

    Nearest neighbour never invents label values (unlike the bicubic resize in [M]).

    Raises:
        ValueError: If the map is not 2-D or the axes would need different scales.
    """
    if labels.ndim != 2:
        raise ValueError(f"label map must be 2-D, got shape {labels.shape}")
    if tuple(labels.shape) == tuple(target_shape):
        return labels.copy()
    _check_uniform_scale(labels.shape, target_shape)
    rows, cols = target_shape
    return cv2.resize(labels, (cols, rows), interpolation=cv2.INTER_NEAREST_EXACT)


def build_scene_ground_truth(
    raw: RawSceneAnnotation, config: GroundTruthConfig | None = None
) -> SceneGroundTruth:
    """Apply the documented GT policy: binarise → area-resize → label BLOBs.

    No morphological opening is applied (D8).

    Args:
        raw: Original annotation arrays.
        config: Preprocessing policy; defaults to :class:`GroundTruthConfig`.

    Returns:
        A :class:`SceneGroundTruth` at image resolution.
    """
    config = config or GroundTruthConfig()
    foreground = resize_binary_mask(
        to_binary_foreground(raw.mask), config.target_shape, config.foreground_threshold
    )
    blob_labels, n_blobs = label_blobs(foreground, config.connectivity)
    class_map = None
    if raw.class_map is not None:
        class_map = resize_label_map(raw.class_map, config.target_shape)
    return SceneGroundTruth(
        scene=raw.scene,
        foreground=foreground,
        blob_labels=blob_labels,
        n_blobs=n_blobs,
        class_map=class_map,
        source_mask_shape=tuple(int(v) for v in raw.mask.shape),
        config=config,
    )


def check_alignment(image: np.ndarray, ground_truth: SceneGroundTruth) -> None:
    """Raise ``ValueError`` if an image and its ground truth differ in spatial shape."""
    if tuple(image.shape[:2]) != tuple(ground_truth.foreground.shape):
        raise ValueError(
            f"Image shape {image.shape[:2]} does not match ground truth "
            f"{ground_truth.foreground.shape} for scene {ground_truth.scene}"
        )
