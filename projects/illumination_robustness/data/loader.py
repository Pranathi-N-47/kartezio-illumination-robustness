"""Image and sample loading without silent rescaling or channel confusion.

Preserves raw numeric values and dtypes until explicit normalisation.
Checks spatial alignment with ground truth and validates single-channel 2-D images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import tifffile

from projects.illumination_robustness.data import benchmark_info as info
from projects.illumination_robustness.data.ground_truth import (
    GroundTruthConfig,
    SceneGroundTruth,
    build_scene_ground_truth,
    check_alignment,
    load_scene_annotation,
)
from projects.illumination_robustness.data.manifest import (
    ManifestRecord,
    parse_benchmark_filename,
)


@dataclass(frozen=True)
class BenchmarkSample:
    """A benchmark image and its corresponding ground truth.

    Attributes:
        image_id: Stable identifier, e.g. ``r01_b02_n03``.
        scene: Scene index r in 1..4.
        shading: Shading index b in 1..13.
        noise: Noise index n in 1..14.
        image: Raw 2-D image array (typically uint8, 1000x1500), unmodified.
        ground_truth: Aligned ground-truth object for the scene.
        image_path: Path to the image file.
    """

    image_id: str
    scene: int
    shading: int
    noise: int
    image: np.ndarray
    ground_truth: SceneGroundTruth
    image_path: Path

    @property
    def mask(self) -> np.ndarray:
        """Convenience property returning the boolean foreground mask."""
        return self.ground_truth.foreground


class BenchmarkDataset:
    """Dataset manager providing access to benchmark images and ground truth.

    Caches the 4 preprocessed scene ground-truth objects in memory to avoid
    re-reading and re-resizing .mat files on every image access.
    """

    def __init__(
        self,
        benchmark_root: Union[str, Path],
        gt_config: Optional[GroundTruthConfig] = None,
        allow_color_conversion: bool = False,
    ) -> None:
        """Initialize the dataset manager.

        Args:
            benchmark_root: Root directory containing ``data/`` and ``truth/``.
            gt_config: Ground truth configuration policy.
            allow_color_conversion: If False (default), reject images with >1 channel.
                If True, convert multi-channel images to grayscale using OpenCV.
        """
        self.benchmark_root = Path(benchmark_root).resolve()
        self.gt_config = gt_config or GroundTruthConfig()
        self.allow_color_conversion = allow_color_conversion
        self._gt_cache: Dict[int, SceneGroundTruth] = {}

    def get_ground_truth(self, scene: int) -> SceneGroundTruth:
        """Get or load cached GroundTruth for the given scene."""
        if scene not in info.SCENES:
            raise ValueError(f"Invalid scene {scene}; expected one of {info.SCENES}")
        if scene not in self._gt_cache:
            raw = load_scene_annotation(self.benchmark_root, scene)
            self._gt_cache[scene] = build_scene_ground_truth(raw, self.gt_config)
        return self._gt_cache[scene]

    def load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load a raw image preserving dtype and values.

        Args:
            path: Path to the TIFF image file.

        Returns:
            2-D NumPy array of the image.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the image cannot be read or has unexpected channels/dimensions.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        try:
            arr = tifffile.imread(file_path)
        except Exception as err:
            raise ValueError(f"Failed to read TIFF image {file_path}: {err}") from err

        arr = np.asarray(arr)
        if arr.ndim == 3:
            if not self.allow_color_conversion:
                raise ValueError(
                    f"Unexpected multi-channel image ({arr.shape}) at {file_path}. "
                    "Expected 2-D grayscale. Set allow_color_conversion=True to convert."
                )
            import cv2
            if arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            elif arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            else:
                raise ValueError(f"Cannot convert image with {arr.shape[2]} channels to grayscale.")
        elif arr.ndim != 2:
            raise ValueError(f"Expected 2-D image array, got shape {arr.shape} for {file_path}")

        return arr

    def load_sample(
        self,
        scene: int,
        shading: int,
        noise: int,
        image_dir: Optional[Union[str, Path]] = None,
    ) -> BenchmarkSample:
        """Load a sample by (scene, shading, noise) coordinates."""
        img_dir = Path(image_dir) if image_dir else self.benchmark_root / info.DATA_SUBDIR
        filename = info.benchmark_filename(scene, shading, noise)
        img_path = img_dir / filename
        return self.load_sample_from_path(img_path)

    def load_sample_from_path(self, path: Union[str, Path]) -> BenchmarkSample:
        """Load a sample given an image file path."""
        p = Path(path)
        scene, shading, noise = parse_benchmark_filename(p.name)
        image = self.load_image(p)
        gt = self.get_ground_truth(scene)
        check_alignment(image, gt)

        return BenchmarkSample(
            image_id=info.image_id(scene, shading, noise),
            scene=scene,
            shading=shading,
            noise=noise,
            image=image,
            ground_truth=gt,
            image_path=p,
        )

    def load_sample_from_record(
        self, record: ManifestRecord, repo_root: Optional[Path] = None
    ) -> BenchmarkSample:
        """Load a sample directly using a ManifestRecord."""
        img_path = Path(record.image_path)
        if not img_path.is_absolute() and repo_root is not None:
            img_path = repo_root / img_path
        return self.load_sample_from_path(img_path)
