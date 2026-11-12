"""Redistribute the PlantDisease YOLO dataset into a stratified split.

The script is intentionally strict: it validates image/label pairing, stages the
new split in a temporary area, and only replaces the current dataset after an
explicit terminal confirmation. The original dataset files are preserved until
all validations pass.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "valid", "test")
SPLIT_INDEX = {"train": 0, "valid": 1, "test": 2}


@dataclass(slots=True, frozen=True)
class SamplePair:
    """Keep an image and its YOLO label together during redistribution."""

    split: str
    stem: str
    class_id: int
    image_path: Path
    label_path: Path


@dataclass(slots=True)
class DatasetInventory:
    """Hold the current dataset state and the proposed stratified split."""

    samples: list[SamplePair]
    original_counts: dict[str, dict[str, int]]
    class_counts: Counter[int]
    missing_images: list[Path]
    missing_labels: list[Path]
    duplicate_image_stems: dict[str, list[Path]]
    duplicate_label_stems: dict[str, list[Path]]


@dataclass(slots=True)
class SplitPlan:
    """Store the target distribution for each class across train, valid, and test."""

    by_class: dict[int, dict[str, list[SamplePair]]]

    def counts_by_class(self) -> dict[int, dict[str, int]]:
        return {
            class_id: {split: len(items) for split, items in split_map.items()}
            for class_id, split_map in self.by_class.items()
        }


class DatasetValidationError(RuntimeError):
    """Raised when the dataset cannot be safely redistributed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redistribute the PlantDisease dataset with stratified sampling."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)

    dataset_root = args.dataset.expanduser().resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_root}")

    inventory = collect_inventory(dataset_root)
    if inventory.missing_images or inventory.missing_labels:
        report_pairing_issues(inventory.missing_images, inventory.missing_labels)
        raise SystemExit("Dataset pairing validation failed. Aborting without changes.")

    if any(inventory.duplicate_image_stems.values()) or any(
        inventory.duplicate_label_stems.values()
    ):
        report_duplicate_issues(
            inventory.duplicate_image_stems, inventory.duplicate_label_stems
        )
        raise SystemExit("Duplicate stem validation failed. Aborting without changes.")

    if not inventory.samples:
        raise SystemExit("Dataset is empty. Aborting without changes.")

    split_plan = build_stratified_plan(
        inventory.samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )

    print_current_summary(inventory)
    print_new_distribution(split_plan, inventory.class_counts)

    confirmation = input("Digite CONFIRMA para aplicar a nova divisão:").strip()
    if confirmation != "CONFIRMA":
        print("Redistribution cancelled. No files were changed.")
        return

    stage_root = stage_new_dataset(dataset_root, split_plan)
    validate_staged_dataset(stage_root, split_plan, inventory)
    apply_staged_dataset(dataset_root, stage_root)

    refreshed_inventory = collect_inventory(dataset_root)
    if refreshed_inventory.missing_images or refreshed_inventory.missing_labels:
        raise DatasetValidationError(
            "Post-redistribution validation found unmatched image/label files."
        )

    print_post_validation(dataset_root)
    print(f"Redistribution completed successfully in: {dataset_root}")


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    total = train_ratio + valid_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise SystemExit(f"Split ratios must sum to 1.0. Received total={total:.6f}.")
    if train_ratio <= 0 or valid_ratio <= 0 or test_ratio <= 0:
        raise SystemExit("Split ratios must all be greater than zero.")


def collect_inventory(dataset_root: Path) -> DatasetInventory:
    samples: list[SamplePair] = []
    original_counts = {split: {"images": 0, "labels": 0} for split in SPLITS}
    class_counts: Counter[int] = Counter()
    missing_images: list[Path] = []
    missing_labels: list[Path] = []
    image_stems: dict[str, list[Path]] = defaultdict(list)
    label_stems: dict[str, list[Path]] = defaultdict(list)

    for split in SPLITS:
        split_root = dataset_root / split
        images_dir = split_root / "images"
        labels_dir = split_root / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            raise DatasetValidationError(
                f"Missing expected split directories under {split_root}."
            )

        images = [
            path
            for path in sorted(images_dir.iterdir())
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        labels = [
            path
            for path in sorted(labels_dir.iterdir())
            if path.is_file() and path.suffix.lower() == ".txt"
        ]

        original_counts[split]["images"] = len(images)
        original_counts[split]["labels"] = len(labels)

        for image_path in images:
            image_stems[image_path.stem].append(image_path)
        for label_path in labels:
            label_stems[label_path.stem].append(label_path)

    all_image_stems = set(image_stems)
    all_label_stems = set(label_stems)

    for stem in sorted(all_image_stems - all_label_stems):
        missing_labels.append(image_stems[stem][0])
    for stem in sorted(all_label_stems - all_image_stems):
        missing_images.append(label_stems[stem][0])

    duplicate_image_stems = {
        stem: paths for stem, paths in image_stems.items() if len(paths) > 1
    }
    duplicate_label_stems = {
        stem: paths for stem, paths in label_stems.items() if len(paths) > 1
    }

    for stem in sorted(all_image_stems & all_label_stems):
        image_path = image_stems[stem][0]
        label_path = label_stems[stem][0]
        class_id = read_label_class_id(label_path)
        split = image_path.parent.parent.name
        samples.append(
            SamplePair(
                split=split,
                stem=stem,
                class_id=class_id,
                image_path=image_path,
                label_path=label_path,
            )
        )
        class_counts[class_id] += 1

    return DatasetInventory(
        samples=samples,
        original_counts=original_counts,
        class_counts=class_counts,
        missing_images=missing_images,
        missing_labels=missing_labels,
        duplicate_image_stems=duplicate_image_stems,
        duplicate_label_stems=duplicate_label_stems,
    )


def read_label_class_id(label_path: Path) -> int:
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise DatasetValidationError(f"Empty label file: {label_path}")
    first_line = lines[0].split()
    if len(first_line) < 1:
        raise DatasetValidationError(f"Invalid label format in file: {label_path}")
    try:
        return int(first_line[0])
    except ValueError as exc:
        raise DatasetValidationError(f"Invalid class id in file: {label_path}") from exc


def build_stratified_plan(
    samples: list[SamplePair],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> SplitPlan:
    rng = random.Random(seed)
    grouped: dict[int, list[SamplePair]] = defaultdict(list)
    for sample in samples:
        grouped[sample.class_id].append(sample)

    by_class: dict[int, dict[str, list[SamplePair]]] = {}
    for class_id in sorted(grouped):
        class_samples = grouped[class_id][:]
        rng.shuffle(class_samples)
        counts = compute_split_counts(
            len(class_samples),
            train_ratio,
            valid_ratio,
            test_ratio,
        )
        by_class[class_id] = {
            "train": class_samples[: counts[0]],
            "valid": class_samples[counts[0] : counts[0] + counts[1]],
            "test": class_samples[counts[0] + counts[1] :],
        }

    return SplitPlan(by_class=by_class)


def compute_split_counts(
    sample_count: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    ratios = [train_ratio, valid_ratio, test_ratio]
    if sample_count == 0:
        return (0, 0, 0)
    if sample_count == 1:
        return (1, 0, 0)
    if sample_count == 2:
        return (1, 1, 0)

    minimum_counts = [1, 1, 1]
    remaining = sample_count - sum(minimum_counts)
    if remaining < 0:
        minimum_counts = [0, 0, 0]
        remaining = sample_count

    desired = [remaining * ratio for ratio in ratios]
    base_counts = [int(math.floor(value)) for value in desired]
    allocated = sum(base_counts)
    leftovers = remaining - allocated
    fractions = [value - int(math.floor(value)) for value in desired]
    tie_break = {0: 0, 1: 1, 2: 2}
    order = sorted(
        range(3),
        key=lambda index: (-fractions[index], tie_break[index]),
    )

    counts = [minimum_counts[index] + base_counts[index] for index in range(3)]
    for index in order[:leftovers]:
        counts[index] += 1

    return tuple(counts)  # type: ignore[return-value]


def stage_new_dataset(dataset_root: Path, split_plan: SplitPlan) -> Path:
    stage_root = Path(tempfile.mkdtemp(prefix="plant_disease_redistribute_stage_"))
    try:
        for split in SPLITS:
            (stage_root / split / "images").mkdir(parents=True, exist_ok=True)
            (stage_root / split / "labels").mkdir(parents=True, exist_ok=True)

        for class_id, split_map in split_plan.by_class.items():
            for split, items in split_map.items():
                for sample in items:
                    image_dest = stage_root / split / "images" / sample.image_path.name
                    label_dest = stage_root / split / "labels" / sample.label_path.name
                    shutil.copy2(sample.image_path, image_dest)
                    shutil.copy2(sample.label_path, label_dest)
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    return stage_root


def validate_staged_dataset(
    stage_root: Path,
    split_plan: SplitPlan,
    inventory: DatasetInventory,
) -> None:
    staged_inventory = collect_inventory(stage_root)
    if len(staged_inventory.samples) != len(inventory.samples):
        raise DatasetValidationError(
            "Staged dataset size differs from the original dataset."
        )

    original_total = sum(inventory.class_counts.values())
    staged_total = sum(staged_inventory.class_counts.values())
    if original_total != staged_total:
        raise DatasetValidationError("Total sample count changed during staging.")

    expected_counts = split_plan.counts_by_class()
    for class_id, counts in expected_counts.items():
        staged_class_counts = count_class_distribution(stage_root, class_id)
        if staged_class_counts != counts:
            raise DatasetValidationError(
                f"Staged counts for class {class_id} do not match the planned distribution."
            )


def count_class_distribution(stage_root: Path, class_id: int) -> dict[str, int]:
    result = {"train": 0, "valid": 0, "test": 0}
    for split in SPLITS:
        labels_dir = stage_root / split / "labels"
        for label_path in labels_dir.iterdir():
            if not label_path.is_file() or label_path.suffix.lower() != ".txt":
                continue
            try:
                label_class_id = read_label_class_id(label_path)
            except DatasetValidationError:
                continue
            if label_class_id == class_id:
                result[split] += 1
    return result


def apply_staged_dataset(dataset_root: Path, stage_root: Path) -> None:
    backup_root = Path(tempfile.mkdtemp(prefix="plant_disease_redistribute_backup_"))
    moved_backups: list[str] = []
    moved_new: list[str] = []
    try:
        for split in SPLITS:
            source = dataset_root / split
            backup_target = backup_root / split
            if source.exists():
                shutil.move(str(source), str(backup_target))
                moved_backups.append(split)

        for split in SPLITS:
            staged_split = stage_root / split
            destination = dataset_root / split
            shutil.move(str(staged_split), str(destination))
            moved_new.append(split)
    except Exception:
        for split in reversed(moved_new):
            destination = dataset_root / split
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
        for split in reversed(moved_backups):
            backup_source = backup_root / split
            destination = dataset_root / split
            if backup_source.exists():
                shutil.move(str(backup_source), str(destination))
        shutil.rmtree(stage_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)
        raise
    else:
        shutil.rmtree(stage_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)


def print_current_summary(inventory: DatasetInventory) -> None:
    print("ANTES:")
    total_images = 0
    total_labels = 0
    for split in SPLITS:
        split_counts = inventory.original_counts[split]
        total_images += split_counts["images"]
        total_labels += split_counts["labels"]
        print(
            f"{split}: images={split_counts['images']} labels={split_counts['labels']}"
        )
    print(f"total: images={total_images} labels={total_labels}")
    print()
    print("CLASSES:")
    for class_id in sorted(inventory.class_counts):
        print(f"class {class_id}: {inventory.class_counts[class_id]}")
    print()


def print_new_distribution(split_plan: SplitPlan, class_counts: Counter[int]) -> None:
    print("NOVA DIVISÃO:")
    header = f"{'':14} {'train':>8} {'valid':>8} {'test':>8} {'total':>8}"
    print(header)
    split_totals = {"train": 0, "valid": 0, "test": 0}
    for class_id in sorted(class_counts):
        counts = split_plan.by_class[class_id]
        train_count = len(counts["train"])
        valid_count = len(counts["valid"])
        test_count = len(counts["test"])
        total = train_count + valid_count + test_count
        split_totals["train"] += train_count
        split_totals["valid"] += valid_count
        split_totals["test"] += test_count
        print(
            f"class {class_id:<8} {train_count:>8} {valid_count:>8} {test_count:>8} {total:>8}"
        )
    print(
        f"TOTAL{'':9} {split_totals['train']:>8} {split_totals['valid']:>8} {split_totals['test']:>8} {sum(split_totals.values()):>8}"
    )
    print()


def print_post_validation(dataset_root: Path) -> None:
    print("PÓS-VALIDAÇÃO:")
    refreshed_inventory = collect_inventory(dataset_root)
    for split in SPLITS:
        split_counts = refreshed_inventory.original_counts[split]
        print(
            f"{split}: images={split_counts['images']} labels={split_counts['labels']}"
        )
    print()
    class_distribution = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        labels_dir = dataset_root / split / "labels"
        for label_path in labels_dir.iterdir():
            if not label_path.is_file() or label_path.suffix.lower() != ".txt":
                continue
            class_id = read_label_class_id(label_path)
            class_distribution[split][class_id] += 1

    print("CLASSES POR SPLIT:")
    for split in SPLITS:
        for class_id in sorted(refreshed_inventory.class_counts):
            count = class_distribution[split][class_id]
            print(f"{split} - class {class_id}: images={count} labels={count}")
    print()


def report_pairing_issues(
    missing_images: Iterable[Path], missing_labels: Iterable[Path]
) -> None:
    if missing_labels:
        print("Imagens sem label correspondente:")
        for path in missing_labels:
            print(f"- {path}")
    if missing_images:
        print("Labels sem imagem correspondente:")
        for path in missing_images:
            print(f"- {path}")


def report_duplicate_issues(
    duplicate_image_stems: dict[str, list[Path]],
    duplicate_label_stems: dict[str, list[Path]],
) -> None:
    if duplicate_image_stems:
        print("Stems de imagem duplicados:")
        for stem, paths in duplicate_image_stems.items():
            print(f"- {stem}")
            for path in paths:
                print(f"  {path}")
    if duplicate_label_stems:
        print("Stems de label duplicados:")
        for stem, paths in duplicate_label_stems.items():
            print(f"- {stem}")
            for path in paths:
                print(f"  {path}")


if __name__ == "__main__":
    try:
        main()
    except DatasetValidationError as exc:
        raise SystemExit(str(exc)) from exc
