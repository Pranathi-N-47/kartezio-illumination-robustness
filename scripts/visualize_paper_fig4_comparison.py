"""Generate visual comparison matching Fig 4 of the paper (PLOS ONE 10(7): e0131098).

Fig 4 in the paper displays:
- A plot of Q vs Artifact level A(r, b, n) for the 13-image series (r=1, b=1..13, n=b+1).
- 5 sampled condition images with arrows pointing to them:
  1. Point 1: b=1, n=2  (A ≈ 0.005) - Clean
  2. Point 4: b=4, n=5  (A ≈ 0.256) - Mild
  3. Point 6: b=6, n=7  (A ≈ 0.555) - Transition
  4. Point 8: b=8, n=9  (A ≈ 0.749) - Otsu collapse
  5. Point 13: b=13, n=14 (A = 1.000) - Extreme
- Row 1: Original input images
- Row 2: Otsu segmentations (red outline)
- Row 3: Sobel edge segmentations (blue outline)
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from projects.illumination_robustness.data.loader import BenchmarkDataset
from projects.illumination_robustness.data import benchmark_info as info
from projects.illumination_robustness.evaluation.baselines import (
    OtsuConfig,
    SobelConfig,
    run_otsu_baseline,
    run_sobel_baseline,
)
from projects.illumination_robustness.evaluation.evaluator import evaluate_prediction
from projects.illumination_robustness.evaluation.metrics import compute_artifact_level


def generate_comparison_figure(
    benchmark_root: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = BenchmarkDataset(benchmark_root=benchmark_root)
    otsu_cfg = OtsuConfig()
    sobel_cfg = SobelConfig()

    # 1. Evaluate all 13 points in the robustness series
    artifacts = []
    otsu_q_segs, sobel_q_segs = [], []
    otsu_dices, sobel_dices = [], []

    # Map of key 5 points corresponding to Fig 4 arrows
    key_indices = [0, 3, 5, 7, 12]  # 0-indexed for 1, 4, 6, 8, 13
    key_samples = []

    print("Evaluating 13 series images...", flush=True)
    for idx, (b, n) in enumerate(info.ROBUSTNESS_SERIES):
        sample = dataset.load_sample(scene=1, shading=b, noise=n)
        a_level = compute_artifact_level(sample.image, noise_index=n)
        artifacts.append(a_level)

        mask_ot = run_otsu_baseline(sample.image, otsu_cfg)
        res_ot = evaluate_prediction(mask_ot, sample.ground_truth, sample.image_id, b, n, 0.0)
        otsu_q_segs.append(res_ot.q_seg)
        otsu_dices.append(res_ot.dice)

        mask_so = run_sobel_baseline(sample.image, sobel_cfg)
        res_so = evaluate_prediction(mask_so, sample.ground_truth, sample.image_id, b, n, 0.0)
        sobel_q_segs.append(res_so.q_seg)
        sobel_dices.append(res_so.dice)

        if idx in key_indices:
            key_samples.append({
                "point_idx": idx + 1,
                "shading": b,
                "noise": n,
                "artifact": a_level,
                "image": sample.image,
                "gt": sample.ground_truth,
                "mask_ot": mask_ot,
                "res_ot": res_ot,
                "mask_so": mask_so,
                "res_so": res_so,
            })

    r_otsu = float(np.mean(otsu_q_segs))
    r_sobel = float(np.mean(sobel_q_segs))

    print(f"Computed R_otsu = {r_otsu:.4f}, R_sobel = {r_sobel:.4f}")

    # 2. Build figure matching Fig 4 layout
    # Top: Robustness curve plot
    # Bottom: 4 rows x 5 columns (Input, GT, Otsu, Sobel)
    fig = plt.figure(figsize=(15, 14), constrained_layout=False)
    gs = fig.add_gridspec(5, 5, height_ratios=[2.2, 1.2, 1.2, 1.2, 1.2], hspace=0.25, wspace=0.15)

    # Plot in top area spanning all 5 columns
    ax_plot = fig.add_subplot(gs[0, :])
    ax_plot.plot(
        artifacts,
        otsu_q_segs,
        "s--",
        color="#cc0000",
        linewidth=2.2,
        markersize=7,
        label=f"Otsu (Python Q_seg, R={r_otsu:.2f}) [Paper R=0.54]",
    )
    ax_plot.plot(
        artifacts,
        sobel_q_segs,
        "s--",
        color="#0055cc",
        linewidth=2.2,
        markersize=7,
        label=f"Sobel Edge (Python Q_seg, R={r_sobel:.2f}) [Paper R=0.47]",
    )
    # Also show Dice as a reference
    ax_plot.plot(
        artifacts,
        otsu_dices,
        ":",
        color="#ff7777",
        linewidth=1.5,
        label="Otsu (Python Dice)",
    )
    ax_plot.plot(
        artifacts,
        sobel_dices,
        ":",
        color="#66aaff",
        linewidth=1.5,
        label="Sobel Edge (Python Dice)",
    )

    ax_plot.set_title(
        "Figure 4 Reproduction: Baseline Quality Q(1, b, n) vs Increasing Artifact Level A(1, b, n)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax_plot.set_xlabel("Artifact Level A(r, b, n) (Shading + Noise)", fontsize=11)
    ax_plot.set_ylabel("Quality Criterion Q(r, b, n)", fontsize=11)
    ax_plot.set_xlim(-0.02, 1.05)
    ax_plot.set_ylim(-0.05, 1.05)
    ax_plot.grid(True, linestyle="--", alpha=0.5)
    ax_plot.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # Annotate the 5 milestone points with vertical lines and markers
    for item in key_samples:
        a_val = item["artifact"]
        ax_plot.axvline(x=a_val, color="gray", linestyle=":", alpha=0.6)
        ax_plot.text(
            a_val,
            -0.03,
            f"P{item['point_idx']}\n(b={item['shading']})",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="semibold",
        )

    # Helper function to crop or show image nicely
    # To see objects clearly like paper Fig 4, we show the full image downscaled
    def display_binary_mask(ax, mask, border_color, title=None):
        # Display binary mask with black objects on white background (matching paper style)
        # In paper Fig 4: white background, objects colored.
        h, w = mask.shape
        rgb = np.ones((h, w, 3), dtype=np.uint8) * 255
        # Objects are foreground (mask > 0)
        rgb[mask > 0] = [139, 35, 35]  # Brown color like paper
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=9)
        # Add colored border rectangle
        rect = patches.Rectangle(
            (0, 0), w - 1, h - 1, linewidth=3, edgecolor=border_color, facecolor="none"
        )
        ax.add_patch(rect)

    # 3. Fill the 4 image rows
    row_labels = ["Input X(1,b,n)", "Ground Truth", "Otsu (Python)", "Sobel (Python)"]

    for col_idx, item in enumerate(key_samples):
        # Row 1: Original Image
        ax_in = fig.add_subplot(gs[1, col_idx])
        ax_in.imshow(item["image"], cmap="gray", vmin=0, vmax=255)
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_title(
            f"b={item['shading']}, n={item['noise']}\nA={item['artifact']:.2f}",
            fontsize=9,
            fontweight="bold",
        )
        if col_idx == 0:
            ax_in.set_ylabel("Original Image", fontsize=10, fontweight="bold")

        # Row 2: Ground Truth
        ax_gt = fig.add_subplot(gs[2, col_idx])
        display_binary_mask(ax_gt, item["gt"].foreground, border_color="green")
        if col_idx == 0:
            ax_gt.set_ylabel("Ground Truth", fontsize=10, fontweight="bold")

        # Row 3: Otsu Segmentation (Red border matching paper)
        ax_ot = fig.add_subplot(gs[3, col_idx])
        display_binary_mask(
            ax_ot,
            item["mask_ot"],
            border_color="#cc0000",
            title=f"Dice: {item['res_ot'].dice:.2f} | Q: {item['res_ot'].q_seg:.2f}",
        )
        if col_idx == 0:
            ax_ot.set_ylabel("Otsu Method", fontsize=10, fontweight="bold")

        # Row 4: Sobel Edge Segmentation (Blue border matching paper)
        ax_so = fig.add_subplot(gs[4, col_idx])
        display_binary_mask(
            ax_so,
            item["mask_so"],
            border_color="#0055cc",
            title=f"Dice: {item['res_so'].dice:.2f} | Q: {item['res_so'].q_seg:.2f}",
        )
        if col_idx == 0:
            ax_so.set_ylabel("Sobel Method", fontsize=10, fontweight="bold")

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Comparison figure saved successfully to: {output_path}")


if __name__ == "__main__":
    benchmark_dir = repo_root / "hardware_items"
    out_file = repo_root / "docs" / "audit_vis" / "paper_fig4_vs_python_comparison.png"
    generate_comparison_figure(benchmark_dir, out_file)
