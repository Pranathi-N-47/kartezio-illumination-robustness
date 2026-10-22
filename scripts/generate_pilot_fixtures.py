"""Generate deterministic synthetic images and masks for the pilot test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def save_fixture(root: Path, name: str, image: np.ndarray, mask: np.ndarray, instances: np.ndarray) -> None:
    tifffile.imwrite(root / f"{name}_input.tif", image.astype(np.uint8))
    tifffile.imwrite(root / f"{name}_ground_truth.tif", (mask.astype(np.uint8) * 255))
    tifffile.imwrite(root / f"{name}_instances.tif", instances.astype(np.uint16))
    cv2.imwrite(str(root / f"{name}_input.png"), image.astype(np.uint8))
    cv2.imwrite(str(root / f"{name}_ground_truth.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(str(root / f"{name}_instances.png"), instances.astype(np.uint16))


def generate(root: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    size = (128, 128)

    def canvas() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.full(size, 220, np.uint8), np.zeros(size, bool), np.zeros(size, np.uint16)

    def add(mask, instances, shape, label):
        object_mask = shape.astype(bool)
        mask |= object_mask
        instances[object_mask] = label

    yy, xx = np.ogrid[:128, :128]

    image, mask, inst = canvas()
    add(mask, inst, (yy - 64) ** 2 + (xx - 64) ** 2 <= 22 ** 2, 1)
    image[mask] = 30
    save_fixture(root, "01_single_disk", image, mask, inst)

    image, mask, inst = canvas()
    add(mask, inst, (yy - 38) ** 2 + (xx - 38) ** 2 <= 14 ** 2, 1)
    add(mask, inst, (yy - 88) ** 2 + (xx - 86) ** 2 <= 18 ** 2, 2)
    image[mask] = 35
    save_fixture(root, "02_multiple_disks", image, mask, inst)

    image, mask, inst = canvas()
    add(mask, inst, (yy >= 20) & (yy < 55) & (xx >= 15) & (xx < 52), 1)
    add(mask, inst, (yy >= 70) & (yy < 108) & (xx >= 74) & (xx < 112), 2)
    image[mask] = 25
    save_fixture(root, "03_rectangles", image, mask, inst)

    image, mask, inst = canvas()
    ellipse = (((yy - 42) / 15) ** 2 + ((xx - 42) / 28) ** 2) <= 1
    triangle = (yy >= 70) & (yy <= 110) & (xx >= 20) & (xx <= 70) & (xx <= 20 + (yy - 70) * 1.25)
    add(mask, inst, ellipse, 1)
    add(mask, inst, triangle, 2)
    image[mask] = 40
    save_fixture(root, "04_ellipse_triangle", image, mask, inst)

    image, mask, inst = canvas()
    add(mask, inst, (yy >= 52) & (yy < 76) & (xx >= 10) & (xx < 116), 1)
    add(mask, inst, (yy >= 10) & (yy < 116) & (xx >= 58) & (xx < 72), 2)
    image[mask] = 30
    save_fixture(root, "05_thin_crossing_objects", image, mask, inst)

    image, mask, inst = canvas()
    left = (yy - 64) ** 2 + (xx - 52) ** 2 <= 20 ** 2
    right = (yy - 64) ** 2 + (xx - 76) ** 2 <= 20 ** 2
    add(mask, inst, left, 1)
    add(mask, inst, right, 2)
    image[mask] = 30
    save_fixture(root, "06_touching_disks", image, mask, inst)

    image, mask, inst = canvas()
    outer = (yy - 64) ** 2 + (xx - 64) ** 2 <= 28 ** 2
    inner = (yy - 64) ** 2 + (xx - 64) ** 2 <= 10 ** 2
    ring = outer & ~inner
    add(mask, inst, ring, 1)
    image[mask] = 25
    save_fixture(root, "07_object_with_hole", image, mask, inst)

    image, mask, inst = canvas()
    add(mask, inst, (yy < 42) & (xx < 44), 1)
    add(mask, inst, (yy > 86) & (xx > 84), 2)
    image[mask] = 35
    save_fixture(root, "08_border_objects", image, mask, inst)

    image, mask, inst = canvas()
    gradient = np.tile(np.linspace(160, 245, 128, dtype=np.float32), (128, 1))
    image = gradient.astype(np.uint8)
    add(mask, inst, (yy - 64) ** 2 + (xx - 64) ** 2 <= 20 ** 2, 1)
    image[mask] = 25
    save_fixture(root, "09_illumination_gradient", image, mask, inst)

    image, mask, inst = canvas()
    noise = rng.normal(0, 12, size).astype(np.float32)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    add(mask, inst, (yy - 42) ** 2 + (xx - 42) ** 2 <= 16 ** 2, 1)
    add(mask, inst, (yy - 88) ** 2 + (xx - 88) ** 2 <= 12 ** 2, 2)
    image[mask] = np.clip(image[mask].astype(np.int16) - 100, 0, 255).astype(np.uint8)
    save_fixture(root, "10_noisy_multiple_objects", image, mask, inst)

    image, mask, inst = canvas()
    save_fixture(root, "11_empty", image, mask, inst)

    image = np.full(size, 128, np.uint8)
    save_fixture(root, "12_constant", image, np.zeros(size, bool), np.zeros(size, np.uint16))

    print(f"Generated 12 pilot fixtures in {root}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilot_fixtures"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
