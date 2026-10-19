"""Deterministic train/test scene and condition splits with leakage prevention.

Implements leave-one-scene-out cross-validation across the 4 benchmark scenes.
Predefines condition subsets for:
  - Quality-only strategy (easy illumination/noise conditions)
  - Robustness-aware strategy (stratified easy, medium, hard conditions)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

from projects.illumination_robustness.data import benchmark_info as info

SplitStrategy = Literal["quality_only", "robustness_aware", "all_conditions"]

# Rule for "Easy" conditions: low shading (bright, low attenuation: b in 1..3)
# and low noise (n in 1..3).
# Total: 3 x 3 = 9 conditions per scene.
EASY_CONDITIONS: Tuple[Tuple[int, int], ...] = tuple(
    (b, n) for b in (1, 2, 3) for n in (1, 2, 3)
)

# Rule for "Stratified Multi-Condition" (Robustness-aware):
# Evenly sampled across shading (easy: 1, medium: 7, hard: 13)
# and noise (easy: 1, medium: 7, hard: 14).
# Total: 3 x 3 = 9 conditions per scene, balanced in size to EASY_CONDITIONS for fair comparison.
STRATIFIED_CONDITIONS_9: Tuple[Tuple[int, int], ...] = (
    (1, 1),   # Easy shading, easy noise
    (1, 7),   # Easy shading, med noise
    (1, 14),  # Easy shading, hard noise
    (7, 1),   # Med shading, easy noise
    (7, 7),   # Med shading, med noise
    (7, 14),  # Med shading, hard noise
    (13, 1),  # Hard shading, easy noise
    (13, 7),  # Hard shading, med noise
    (13, 14), # Hard shading, hard noise
)

# Extended stratified condition set (5x5 = 25 conditions) if larger training budget is configured:
STRATIFIED_CONDITIONS_25: Tuple[Tuple[int, int], ...] = tuple(
    (b, n) for b in (1, 4, 7, 10, 13) for n in (1, 4, 7, 11, 14)
)


@dataclass(frozen=True)
class ExperimentSplit:
    """Structured record of train/test splits for an experiment run.

    Attributes:
        split_id: Unique identifier for the split (e.g. ``loso_test_s1_robust``).
        strategy: Strategy name (``quality_only`` or ``robustness_aware``).
        test_scene: The single held-out test scene (1..4).
        train_scenes: The 3 scenes used for training.
        condition_strategy: Name of condition selection strategy.
        train_conditions: List of (shading, noise) tuples used for training.
        train_image_ids: Full list of training image IDs.
        test_image_ids: Full list of held-out test image IDs (all 182 conditions).
        test_robustness_series_ids: The 13 image IDs forming the paper's robustness curve.
    """

    split_id: str
    strategy: SplitStrategy
    test_scene: int
    train_scenes: List[int]
    condition_strategy: str
    train_conditions: List[Tuple[int, int]]
    train_image_ids: List[str]
    test_image_ids: List[str]
    test_robustness_series_ids: List[str]

    def validate(self) -> None:
        """Verify absence of data leakage and coverage sanity.

        Raises:
            ValueError: If test images leak into training or conditions are invalid.
        """
        if self.test_scene in self.train_scenes:
            raise ValueError(
                f"Data leakage: test_scene {self.test_scene} is in train_scenes {self.train_scenes}"
            )
        train_set = set(self.train_image_ids)
        test_set = set(self.test_image_ids)
        intersection = train_set.intersection(test_set)
        if intersection:
            raise ValueError(f"Data leakage detected! {len(intersection)} images in both train and test sets.")

        # Ensure all train images belong to train scenes
        for img_id in self.train_image_ids:
            scene_str = img_id.split("_")[0][1:]  # e.g. "r01" -> "01"
            scene_val = int(scene_str)
            if scene_val not in self.train_scenes:
                raise ValueError(f"Image {img_id} belongs to scene {scene_val}, not in train_scenes {self.train_scenes}")

        # Ensure all test images belong to test scene
        for img_id in self.test_image_ids:
            scene_str = img_id.split("_")[0][1:]
            scene_val = int(scene_str)
            if scene_val != self.test_scene:
                raise ValueError(f"Test image {img_id} belongs to scene {scene_val}, not test_scene {self.test_scene}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert split definition to a JSON-serializable dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save split definition to JSON file (UTF-8)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ExperimentSplit:
        """Load split definition from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        # Convert condition lists back to tuples
        train_conds = [tuple(c) for c in data["train_conditions"]]
        data["train_conditions"] = train_conds
        split = cls(**data)
        split.validate()
        return split


def create_loso_split(
    test_scene: int,
    strategy: SplitStrategy = "quality_only",
    custom_conditions: Sequence[Tuple[int, int]] | None = None,
) -> ExperimentSplit:
    """Create a leave-one-scene-out split.

    Args:
        test_scene: Held-out scene r in 1..4.
        strategy: 'quality_only', 'robustness_aware', or 'all_conditions'.
        custom_conditions: Explicit list of (shading, noise) tuples to override defaults.

    Returns:
        Validated :class:`ExperimentSplit`.
    """
    if test_scene not in info.SCENES:
        raise ValueError(f"Invalid test_scene {test_scene}; must be in {info.SCENES}")

    train_scenes = sorted(s for s in info.SCENES if s != test_scene)

    if custom_conditions is not None:
        train_conditions = list(custom_conditions)
        cond_name = f"custom_{len(train_conditions)}"
    elif strategy == "quality_only":
        train_conditions = list(EASY_CONDITIONS)
        cond_name = "easy_9"
    elif strategy == "robustness_aware":
        train_conditions = list(STRATIFIED_CONDITIONS_9)
        cond_name = "stratified_9"
    elif strategy == "all_conditions":
        train_conditions = [
            (b, n) for b in info.SHADING_LEVELS for n in info.NOISE_LEVELS
        ]
        cond_name = "all_182"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Build image IDs
    train_image_ids = [
        info.image_id(s, b, n)
        for s in train_scenes
        for (b, n) in train_conditions
    ]

    test_image_ids = [
        info.image_id(test_scene, b, n)
        for b in info.SHADING_LEVELS
        for n in info.NOISE_LEVELS
    ]

    robustness_ids = [
        info.image_id(test_scene, b, n)
        for (b, n) in info.ROBUSTNESS_SERIES
    ]

    split_id = f"loso_test_s{test_scene}_{strategy}_{cond_name}"
    split = ExperimentSplit(
        split_id=split_id,
        strategy=strategy,
        test_scene=test_scene,
        train_scenes=train_scenes,
        condition_strategy=cond_name,
        train_conditions=train_conditions,
        train_image_ids=train_image_ids,
        test_image_ids=test_image_ids,
        test_robustness_series_ids=robustness_ids,
    )
    split.validate()
    return split
