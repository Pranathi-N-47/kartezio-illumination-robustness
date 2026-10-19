"""Scan the benchmark directory, parse image IDs and validate coverage.

The manifest is the single source of truth for which images exist. Nothing is
silently skipped: every unparseable, duplicated, unreadable or unannotated file is
reported as a :class:`ManifestIssue`, and every expected ``(scene, shading, noise)``
tuple that is absent is reported as ``missing_tuple``.

Annotation naming rule (inspected 2026-10-19, see ``docs/implementation_decisions.md``):
ground truth is per scene, ``<root>/truth/benchdata_{scene}_gd.mat`` (binary mask)
and ``<root>/truth/benchdata_{scene}_class.mat`` (class map, audit only).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import tifffile

from projects.illumination_robustness.data import benchmark_info as info

logger = logging.getLogger(__name__)

# [P, p.3]: 'benchdata_r_b_n.tif', each number written with two digits. The extension
# is matched case-insensitively and '.tiff' is accepted so that a second copy of the
# same key with a different extension is detected as a duplicate rather than ignored.
FILENAME_PATTERN = re.compile(r"^benchdata_(\d{2})_(\d{2})_(\d{2})\.tiff?$", re.IGNORECASE)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",
    "scene",
    "shading",
    "noise",
    "image_path",
    "height",
    "width",
    "dtype",
    "channels",
    "sha256",
    "gt_mask_path",
    "gt_class_path",
    "status",
    "error",
)

# Issue kinds that do not invalidate the manifest for segmentation-only work.
WARNING_ISSUE_KINDS: frozenset[str] = frozenset({"missing_class_map"})


@dataclass
class ManifestRecord:
    """One discovered image file.

    Attributes:
        image_id: Stable ID ``r{RR}_b{BB}_n{NN}``.
        scene: Scene index r.
        shading: Shading index b.
        noise: Noise index n.
        image_path: POSIX path, repository-relative when the file is inside the repo.
        height: Image rows, or ``None`` if unreadable.
        width: Image columns, or ``None`` if unreadable.
        dtype: NumPy dtype name, or ``""`` if unreadable.
        channels: 1 for 2-D images, last-axis length for 3-D, ``None`` if unreadable.
        sha256: Hex SHA-256 of the file bytes.
        gt_mask_path: Path of the scene's binary GT mask file, ``""`` if missing.
        gt_class_path: Path of the scene's class-map file, ``""`` if missing.
        status: ``ok``, ``duplicate``, ``unreadable``, ``unexpected_format`` or
            ``missing_annotation``.
        error: Short human-readable reason when ``status != "ok"``.
    """

    image_id: str
    scene: int
    shading: int
    noise: int
    image_path: str
    height: int | None = None
    width: int | None = None
    dtype: str = ""
    channels: int | None = None
    sha256: str = ""
    gt_mask_path: str = ""
    gt_class_path: str = ""
    status: str = "ok"
    error: str = ""


@dataclass(frozen=True)
class ManifestIssue:
    """A validation problem found while scanning.

    Attributes:
        kind: Machine-readable category (e.g. ``missing_tuple``).
        path: Affected file path, or ``""`` for missing items.
        detail: Human-readable explanation.
    """

    kind: str
    path: str
    detail: str

    @property
    def is_error(self) -> bool:
        """Whether this issue blocks use of the manifest."""
        return self.kind not in WARNING_ISSUE_KINDS


@dataclass
class ManifestResult:
    """Result of :func:`scan_benchmark`."""

    benchmark_root: str
    records: list[ManifestRecord] = field(default_factory=list)
    issues: list[ManifestIssue] = field(default_factory=list)
    n_files_scanned: int = 0

    def summary(self) -> dict:
        """Return a JSON-serialisable validation summary."""
        expected = info.N_SCENES * info.N_SHADING * info.N_NOISE
        issue_counts = Counter(issue.kind for issue in self.issues)
        status_counts = Counter(record.status for record in self.records)
        return {
            "benchmark_root": self.benchmark_root,
            "n_files_scanned": self.n_files_scanned,
            "n_records": len(self.records),
            "n_expected_tuples": expected,
            "n_ok_records": status_counts.get("ok", 0),
            "record_status_counts": dict(sorted(status_counts.items())),
            "issue_counts": dict(sorted(issue_counts.items())),
            "scenes_found": sorted({record.scene for record in self.records}),
            "passed": not any(issue.is_error for issue in self.issues),
        }


def match_benchmark_filename(name: str) -> tuple[int, int, int] | None:
    """Apply the strict filename pattern without range checking.

    Args:
        name: File name (not a path).

    Returns:
        ``(scene, shading, noise)`` if the pattern matches, else ``None``.
    """
    match = FILENAME_PATTERN.fullmatch(name)
    if match is None:
        return None
    scene, shading, noise = (int(group) for group in match.groups())
    return scene, shading, noise


def parse_benchmark_filename(name: str) -> tuple[int, int, int]:
    """Parse and range-check a benchmark image filename.

    Args:
        name: File name such as ``benchdata_02_01_03.tif``.

    Returns:
        ``(scene, shading, noise)`` as integers.

    Raises:
        ValueError: If the name does not match the pattern or an index is outside
            the benchmark's documented range.
    """
    parsed = match_benchmark_filename(name)
    if parsed is None:
        raise ValueError(
            f"{name!r} does not match 'benchdata_RR_BB_NN.tif' (two digits each)"
        )
    problems = _range_problems(*parsed)
    if problems:
        raise ValueError(f"{name!r}: " + "; ".join(problems))
    return parsed


def _range_problems(scene: int, shading: int, noise: int) -> list[str]:
    problems = []
    if scene not in info.SCENES:
        problems.append(f"scene {scene} not in 1..{info.N_SCENES}")
    if shading not in info.SHADING_LEVELS:
        problems.append(f"shading {shading} not in 1..{info.N_SHADING}")
    if noise not in info.NOISE_LEVELS:
        problems.append(f"noise {noise} not in 1..{info.N_NOISE}")
    return problems


def display_path(path: Path, repo_root: Path | None) -> str:
    """Return a POSIX path, relative to ``repo_root`` when ``path`` is inside it."""
    resolved = path.resolve()
    if repo_root is not None:
        try:
            return resolved.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def scene_annotation_paths(benchmark_root: Path, scene: int) -> tuple[Path, Path]:
    """Return ``(mask_path, class_map_path)`` for a scene by the documented naming rule."""
    truth_dir = benchmark_root / info.TRUTH_SUBDIR
    return (
        truth_dir / info.GT_MASK_FILENAME.format(scene=scene),
        truth_dir / info.GT_CLASS_FILENAME.format(scene=scene),
    )


def scan_benchmark(benchmark_root: Path, repo_root: Path | None = None) -> ManifestResult:
    """Scan ``<benchmark_root>/data`` and build a validated manifest.

    Args:
        benchmark_root: Directory containing ``data/`` and ``truth/``
            (e.g. ``hardware_items``).
        repo_root: If given, saved paths are made relative to it when possible.

    Returns:
        A :class:`ManifestResult` with one record per parseable image file and an
        issue for every problem found.

    Raises:
        FileNotFoundError: If ``<benchmark_root>/data`` does not exist.
    """
    data_dir = benchmark_root / info.DATA_SUBDIR
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Benchmark image directory not found: {data_dir}. Expected the "
            f"distributed 'hardware_items' layout with a '{info.DATA_SUBDIR}/' folder."
        )

    result = ManifestResult(benchmark_root=display_path(benchmark_root, repo_root))
    annotation_cache: dict[int, tuple[str, str]] = {}

    for path in sorted(p for p in data_dir.iterdir() if p.is_file()):
        result.n_files_scanned += 1
        shown = display_path(path, repo_root)
        parsed = match_benchmark_filename(path.name)
        if parsed is None:
            result.issues.append(
                ManifestIssue("unparseable_filename", shown, "name does not match pattern")
            )
            continue
        problems = _range_problems(*parsed)
        if problems:
            result.issues.append(ManifestIssue("index_out_of_range", shown, "; ".join(problems)))
            continue

        scene, shading, noise = parsed
        record = ManifestRecord(
            image_id=info.image_id(scene, shading, noise),
            scene=scene,
            shading=shading,
            noise=noise,
            image_path=shown,
        )
        _fill_image_metadata(record, path, result.issues)

        if scene not in annotation_cache:
            annotation_cache[scene] = _check_scene_annotations(
                benchmark_root, scene, repo_root, result.issues
            )
        record.gt_mask_path, record.gt_class_path = annotation_cache[scene]
        if not record.gt_mask_path and record.status == "ok":
            record.status = "missing_annotation"
            record.error = "scene ground-truth mask file not found"
        result.records.append(record)

    _flag_duplicates(result)
    _flag_missing_tuples(result)
    result.records.sort(key=lambda r: (r.scene, r.shading, r.noise, r.image_path))
    logger.info("Scanned %d files: %s", result.n_files_scanned, result.summary())
    return result


def _fill_image_metadata(
    record: ManifestRecord, path: Path, issues: list[ManifestIssue]
) -> None:
    """Read the file once for checksum and array metadata; record failures."""
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        record.status, record.error = "unreadable", f"{type(error).__name__}: {error}"
        issues.append(ManifestIssue("unreadable_file", record.image_path, record.error))
        return
    record.sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        array = tifffile.imread(io.BytesIO(raw_bytes))
    except (tifffile.TiffFileError, ValueError, OSError) as error:
        record.status, record.error = "unreadable", f"{type(error).__name__}: {error}"
        issues.append(ManifestIssue("unreadable_file", record.image_path, record.error))
        return

    array = np.asarray(array)
    record.dtype = array.dtype.name
    if array.ndim == 2:
        record.height, record.width = (int(v) for v in array.shape)
        record.channels = 1
    elif array.ndim == 3:
        record.height, record.width = int(array.shape[0]), int(array.shape[1])
        record.channels = int(array.shape[2])
    else:
        record.status, record.error = "unexpected_format", f"array has ndim={array.ndim}"
        issues.append(ManifestIssue("unexpected_format", record.image_path, record.error))
        return

    reasons = []
    if (record.height, record.width) != info.IMAGE_SHAPE:
        reasons.append(f"shape {(record.height, record.width)} != {info.IMAGE_SHAPE}")
    if record.channels != 1:
        reasons.append(f"{record.channels} channels (grayscale expected)")
    if record.dtype != "uint8":
        reasons.append(f"dtype {record.dtype} (uint8 expected [P])")
    if reasons:
        record.status, record.error = "unexpected_format", "; ".join(reasons)
        issues.append(ManifestIssue("unexpected_format", record.image_path, record.error))


def _check_scene_annotations(
    benchmark_root: Path, scene: int, repo_root: Path | None, issues: list[ManifestIssue]
) -> tuple[str, str]:
    mask_path, class_path = scene_annotation_paths(benchmark_root, scene)
    mask_shown = class_shown = ""
    if mask_path.is_file():
        mask_shown = display_path(mask_path, repo_root)
    else:
        issues.append(
            ManifestIssue(
                "missing_annotation",
                display_path(mask_path, repo_root),
                f"binary ground-truth mask for scene {scene} not found",
            )
        )
    if class_path.is_file():
        class_shown = display_path(class_path, repo_root)
    else:
        issues.append(
            ManifestIssue(
                "missing_class_map",
                display_path(class_path, repo_root),
                f"class map for scene {scene} not found (audit only; not needed for segmentation)",
            )
        )
    return mask_shown, class_shown


def _flag_duplicates(result: ManifestResult) -> None:
    by_key: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in result.records:
        by_key[record.image_id].append(record)
    for image_id, group in by_key.items():
        if len(group) < 2:
            continue
        paths = ", ".join(r.image_path for r in group)
        # All copies are marked: there is no principled way to pick the "right" one.
        for record in group:
            record.status, record.error = "duplicate", f"key {image_id} also at: {paths}"
        result.issues.append(ManifestIssue("duplicate_key", paths, f"{len(group)} files map to {image_id}"))


def _flag_missing_tuples(result: ManifestResult) -> None:
    found = {(r.scene, r.shading, r.noise) for r in result.records}
    for scene in info.SCENES:
        for shading in info.SHADING_LEVELS:
            for noise in info.NOISE_LEVELS:
                if (scene, shading, noise) not in found:
                    result.issues.append(
                        ManifestIssue(
                            "missing_tuple",
                            "",
                            f"{info.benchmark_filename(scene, shading, noise)} not found",
                        )
                    )


def write_manifest(records: list[ManifestRecord], path: Path) -> None:
    """Write manifest records to CSV with stable column order (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            writer.writerow({key: "" if row[key] is None else row[key] for key in MANIFEST_COLUMNS})


def read_manifest(path: Path) -> list[ManifestRecord]:
    """Read a manifest CSV written by :func:`write_manifest`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the header does not match :data:`MANIFEST_COLUMNS`.
    """
    int_fields = {"scene", "shading", "noise"}
    optional_int_fields = {"height", "width", "channels"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(
                f"Unexpected manifest columns in {path}: {reader.fieldnames}; "
                f"expected {list(MANIFEST_COLUMNS)}"
            )
        records = []
        for row in reader:
            kwargs: dict = {}
            for f in fields(ManifestRecord):
                value = row[f.name]
                if f.name in int_fields:
                    kwargs[f.name] = int(value)
                elif f.name in optional_int_fields:
                    kwargs[f.name] = int(value) if value != "" else None
                else:
                    kwargs[f.name] = value
            records.append(ManifestRecord(**kwargs))
    return records


def write_validation_summary(result: ManifestResult, path: Path) -> None:
    """Write the summary plus the full issue list as JSON (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.summary()
    payload["issues"] = [asdict(issue) for issue in result.issues]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
