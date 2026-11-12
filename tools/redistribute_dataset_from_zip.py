"""Rebuild the PlantDisease YOLO dataset from the original ZIP archive.

This script is intentionally self-contained and conservative:
- it reads the archive directly with zipfile;
- it ignores orphan files instead of guessing pairs;
- it stages the new dataset in a temporary directory first;
- it asks for explicit confirmation before writing anything;
- it aborts if the target output directory already exists.

The goal is to reconstruct a clean stratified dataset without modifying the
current dataset or the source ZIP archive.
"""

from __future__ import annotations

import argparse
import ast
import math
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "valid", "test")
SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}
CLASS_NAMES = ("mildew", "rose_P01", "rose_R02")
EXPECTED_CLASS_IDS = {0, 1, 2}
OUTPUT_ROOT_NAME = "dataset_redistributed"
CHUNK_SIZE = 1024 * 1024


@dataclass(slots=True, frozen=True)
class SamplePair:
    """Represent a validated image/label pair from the archive."""

    split: str
    stem: str
    class_id: int
    image_member: str
    label_member: str


@dataclass(slots=True)
class SplitInventory:
    """Track the raw contents and the valid pairs per split."""

    image_members: dict[str, ZipInfo]
    label_members: dict[str, ZipInfo]
    image_stems: Counter[str]
    label_stems: Counter[str]
    valid_pairs: list[SamplePair]
    orphan_images: list[str]
    orphan_labels: list[str]
    duplicate_images: dict[str, list[str]]
    duplicate_labels: dict[str, list[str]]
    class_counts: Counter[int]


@dataclass(slots=True)
class ArchiveAnalysis:
    """Hold the complete diagnosis of the source archive."""

    zip_path: Path
    data_yaml: dict[str, object]
    split_inventory: dict[str, SplitInventory]
    global_valid_pairs: list[SamplePair]
    global_class_counts: Counter[int]
    total_orphan_images: int
    total_orphan_labels: int
    duplicate_full_paths: list[str]


@dataclass(slots=True)
class SplitPlan:
    """Store the deterministic class-balanced split plan."""

    by_class: dict[int, dict[str, list[SamplePair]]]

    def split_counts(self) -> dict[str, Counter[int]]:
        counts = {split: Counter() for split in SPLITS}
        for class_id, split_map in self.by_class.items():
            for split, items in split_map.items():
                counts[split][class_id] = len(items)
        return counts

    def split_totals(self) -> dict[str, int]:
        return {
            split: sum(len(items) for items in split_map.values())
            for split, split_map in self.by_class.items()
        }


class ArchiveIntegrityError(RuntimeError):
    """Raised when the archive cannot be safely redistributed."""


class RedistributionAbort(RuntimeError):
    """Raised when the user or the filesystem refuses the operation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the PlantDisease dataset from the original ZIP archive."
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("dataset") / "archive (1).zip",
        help="Path to the original dataset ZIP archive.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(OUTPUT_ROOT_NAME),
        help="Directory where the redistributed dataset will be created.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)

    zip_path = args.zip_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not zip_path.exists():
        raise SystemExit(f"ZIP archive not found: {zip_path}")
    if output_root.exists():
        raise SystemExit(
            f"Output directory already exists: {output_root}. Aborting for safety."
        )

    analysis = analyze_archive(zip_path)
    if analysis.duplicate_full_paths:
        raise SystemExit(
            "The ZIP archive contains duplicate full paths. Aborting before any copy."
        )

    if any(
        split_inventory.duplicate_images or split_inventory.duplicate_labels
        for split_inventory in analysis.split_inventory.values()
    ):
        report_duplicate_stems(analysis.split_inventory)
        raise SystemExit(
            "Duplicate name-base entries were found in the ZIP. Aborting before any copy."
        )

    if analysis.global_class_counts and not EXPECTED_CLASS_IDS.issubset(
        set(analysis.global_class_counts)
    ):
        raise SystemExit(
            "The ZIP does not contain all required class IDs 0, 1 and 2 in the valid pairs."
        )

    if not analysis.global_valid_pairs:
        raise SystemExit("No valid image/label pairs were found in the ZIP.")

    split_plan = build_stratified_plan(
        analysis.global_valid_pairs,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )

    print_preview(analysis, split_plan)

    confirmation = input("Digite CONFIRMA para criar o novo dataset:").strip()
    if confirmation != "CONFIRMA":
        print("Operation cancelled. No files were created.")
        return

    stage_root = create_staging_dataset(zip_path, output_root, analysis, split_plan)
    try:
        validate_staged_dataset(stage_root, analysis, split_plan)
        stage_to_final_output(stage_root, output_root)
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise

    final_report = analyze_redistributed_output(output_root)
    print_final_validation(final_report)
    print(f"Redistributed dataset created at: {output_root}")


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    total = train_ratio + valid_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise SystemExit(f"Split ratios must sum to 1.0. Received total={total:.6f}.")
    if train_ratio <= 0 or valid_ratio <= 0 or test_ratio <= 0:
        raise SystemExit("Split ratios must all be greater than zero.")


def analyze_archive(zip_path: Path) -> ArchiveAnalysis:
    with ZipFile(zip_path, "r") as archive:
        member_names = [
            info.filename for info in archive.infolist() if not info.is_dir()
        ]
        duplicate_full_paths = [
            name for name, count in Counter(member_names).items() if count > 1
        ]

        data_yaml = read_data_yaml(archive)
        split_inventory: dict[str, SplitInventory] = {
            split: SplitInventory(
                image_members={},
                label_members={},
                image_stems=Counter(),
                label_stems=Counter(),
                valid_pairs=[],
                orphan_images=[],
                orphan_labels=[],
                duplicate_images={},
                duplicate_labels={},
                class_counts=Counter(),
            )
            for split in SPLITS
        }

        for member_name in member_names:
            split, kind = classify_member(member_name)
            if split is None or kind is None:
                continue

            inventory = split_inventory[split]
            member_info = archive.getinfo(member_name)
            stem = PurePosixPath(member_name).stem

            if kind == "image":
                inventory.image_members[stem] = member_info
                inventory.image_stems[stem] += 1
            else:
                inventory.label_members[stem] = member_info
                inventory.label_stems[stem] += 1

        global_valid_pairs: list[SamplePair] = []
        global_class_counts: Counter[int] = Counter()
        total_orphan_images = 0
        total_orphan_labels = 0

        for split in SPLITS:
            inventory = split_inventory[split]
            image_stems = set(inventory.image_members)
            label_stems = set(inventory.label_members)

            inventory.duplicate_images = {
                stem: [
                    member.name
                    for member in archive.infolist()
                    if member.filename == inventory.image_members[stem].filename
                ]
                for stem, count in inventory.image_stems.items()
                if count > 1
            }
            inventory.duplicate_labels = {
                stem: [
                    member.name
                    for member in archive.infolist()
                    if member.filename == inventory.label_members[stem].filename
                ]
                for stem, count in inventory.label_stems.items()
                if count > 1
            }

            orphan_image_stems = sorted(image_stems - label_stems)
            orphan_label_stems = sorted(label_stems - image_stems)
            inventory.orphan_images = [
                inventory.image_members[stem].filename for stem in orphan_image_stems
            ]
            inventory.orphan_labels = [
                inventory.label_members[stem].filename for stem in orphan_label_stems
            ]
            total_orphan_images += len(inventory.orphan_images)
            total_orphan_labels += len(inventory.orphan_labels)

            for stem in sorted(image_stems & label_stems):
                image_member = inventory.image_members[stem]
                label_member = inventory.label_members[stem]
                class_id = read_label_class_id(archive, label_member.filename)
                if class_id not in EXPECTED_CLASS_IDS:
                    raise ArchiveIntegrityError(
                        f"Invalid class id {class_id} found in label: {label_member.filename}"
                    )
                pair = SamplePair(
                    split=split,
                    stem=stem,
                    class_id=class_id,
                    image_member=image_member.filename,
                    label_member=label_member.filename,
                )
                inventory.valid_pairs.append(pair)
                inventory.class_counts[class_id] += 1
                global_valid_pairs.append(pair)
                global_class_counts[class_id] += 1

        if data_yaml is None:
            raise ArchiveIntegrityError("data.yaml was not found in the ZIP archive.")

        return ArchiveAnalysis(
            zip_path=zip_path,
            data_yaml=data_yaml,
            split_inventory=split_inventory,
            global_valid_pairs=global_valid_pairs,
            global_class_counts=global_class_counts,
            total_orphan_images=total_orphan_images,
            total_orphan_labels=total_orphan_labels,
            duplicate_full_paths=duplicate_full_paths,
        )


def classify_member(member_name: str) -> tuple[str | None, str | None]:
    path = PurePosixPath(member_name)
    parts = path.parts
    if len(parts) < 3:
        return None, None
    split = parts[0]
    folder = parts[1]
    if split not in SPLITS:
        return None, None
    if folder == "images" and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
        return split, "image"
    if folder == "labels" and path.suffix.lower() == ".txt":
        return split, "label"
    return None, None


def read_data_yaml(archive: ZipFile) -> dict[str, object] | None:
    yaml_candidates = [
        name
        for name in archive.namelist()
        if PurePosixPath(name).name in {"data.yaml", "data.yml"}
    ]
    if not yaml_candidates:
        return None
    with archive.open(yaml_candidates[0], "r") as handle:
        text = handle.read().decode("utf-8")
    return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict[str, object]:
    result: dict[str, object] = {
        "nc": None,
        "names": None,
        "train": None,
        "val": None,
        "test": None,
    }
    current_key: str | None = None
    collected_names: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("nc:"):
            value = line.split(":", 1)[1].strip()
            try:
                result["nc"] = int(value)
            except ValueError:
                result["nc"] = value
            current_key = None
            continue
        if line.startswith("train:"):
            result["train"] = line.split(":", 1)[1].strip()
            current_key = None
            continue
        if line.startswith("val:"):
            result["val"] = line.split(":", 1)[1].strip()
            current_key = None
            continue
        if line.startswith("test:"):
            result["test"] = line.split(":", 1)[1].strip()
            current_key = None
            continue
        if line.startswith("names:"):
            current_key = "names"
            remainder = line.split(":", 1)[1].strip()
            if remainder:
                try:
                    result["names"] = ast.literal_eval(remainder)
                except Exception:
                    result["names"] = remainder
            continue
        if current_key == "names":
            if line.startswith("-"):
                collected_names.append(line[1:].strip().strip("'\""))
                continue
            if ":" in line:
                index_text, name_text = line.split(":", 1)
                try:
                    index = int(index_text.strip())
                except ValueError:
                    continue
                while len(collected_names) <= index:
                    collected_names.append("")
                collected_names[index] = name_text.strip().strip("'\"")

    if result["names"] is None and collected_names:
        result["names"] = collected_names
    return result


def read_label_class_id(archive: ZipFile, label_member_name: str) -> int:
    with archive.open(label_member_name, "r") as handle:
        lines = [
            line.strip()
            for line in handle.read().decode("utf-8").splitlines()
            if line.strip()
        ]
    if not lines:
        raise ArchiveIntegrityError(f"Empty label file: {label_member_name}")
    first_line = lines[0].split()
    if not first_line:
        raise ArchiveIntegrityError(f"Invalid label content in: {label_member_name}")
    try:
        return int(first_line[0])
    except ValueError as exc:
        raise ArchiveIntegrityError(
            f"Invalid class id in label: {label_member_name}"
        ) from exc


def build_stratified_plan(
    pairs: list[SamplePair],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> SplitPlan:
    rng = random.Random(seed)
    grouped: dict[int, list[SamplePair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.class_id].append(pair)

    plan: dict[int, dict[str, list[SamplePair]]] = {}
    for class_id in sorted(grouped):
        class_pairs = grouped[class_id][:]
        rng.shuffle(class_pairs)
        counts = allocate_counts(len(class_pairs), train_ratio, valid_ratio, test_ratio)
        train_end = counts["train"]
        valid_end = train_end + counts["valid"]
        plan[class_id] = {
            "train": class_pairs[:train_end],
            "valid": class_pairs[train_end:valid_end],
            "test": class_pairs[valid_end:],
        }
    return SplitPlan(by_class=plan)


def allocate_counts(
    sample_count: int, train_ratio: float, valid_ratio: float, test_ratio: float
) -> dict[str, int]:
    ratios = [train_ratio, valid_ratio, test_ratio]
    if sample_count == 0:
        return {"train": 0, "valid": 0, "test": 0}
    if sample_count == 1:
        return {"train": 1, "valid": 0, "test": 0}
    if sample_count == 2:
        return {"train": 1, "valid": 1, "test": 0}

    minimum = [1, 1, 1]
    remaining = sample_count - sum(minimum)
    desired = [remaining * ratio for ratio in ratios]
    base = [math.floor(value) for value in desired]
    leftover = remaining - sum(base)
    fractional = [value - math.floor(value) for value in desired]
    order = sorted(
        range(3), key=lambda index: (-fractional[index], SPLIT_ORDER[SPLITS[index]])
    )

    counts = [minimum[index] + int(base[index]) for index in range(3)]
    for index in order[:leftover]:
        counts[index] += 1

    return {"train": counts[0], "valid": counts[1], "test": counts[2]}


def print_preview(analysis: ArchiveAnalysis, plan: SplitPlan) -> None:
    print("=" * 50)
    print("DATASET REDISTRIBUTION PREVIEW")
    print("=" * 50)
    print()
    print(f"Source:\n{analysis.zip_path}")
    print()
    print(f"Valid pairs:\n{len(analysis.global_valid_pairs)}")
    print()
    print(f"Ignored orphan images:\n{analysis.total_orphan_images}")
    print()
    print(f"Ignored orphan labels:\n{analysis.total_orphan_labels}")
    print()
    print("Class distribution:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"class {class_id} ({class_name}): {analysis.global_class_counts[class_id]}"
        )
    print()
    print("data.yaml inside ZIP:")
    print(f"nc: {analysis.data_yaml.get('nc')}")
    print(f"names: {analysis.data_yaml.get('names')}")
    print(f"train: {analysis.data_yaml.get('train')}")
    print(f"val: {analysis.data_yaml.get('val')}")
    print(f"test: {analysis.data_yaml.get('test')}")
    print()
    print("Proposed split:")
    print_split_table(plan)
    print("=" * 50)
    print()


def print_split_table(plan: SplitPlan) -> None:
    counts = plan.split_counts()
    widths = {"label": 14, "train": 8, "valid": 8, "test": 8, "total": 8}
    print(
        f"{'':<{widths['label']}}{'train':>{widths['train']}}{'valid':>{widths['valid']}}{'test':>{widths['test']}}{'total':>{widths['total']}}"
    )
    total_by_split = Counter()
    for class_id in sorted(plan.by_class):
        class_label = f"class {class_id}"
        train_count = counts["train"][class_id]
        valid_count = counts["valid"][class_id]
        test_count = counts["test"][class_id]
        class_total = train_count + valid_count + test_count
        total_by_split["train"] += train_count
        total_by_split["valid"] += valid_count
        total_by_split["test"] += test_count
        print(
            f"{class_label:<{widths['label']}}"
            f"{train_count:>{widths['train']}}"
            f"{valid_count:>{widths['valid']}}"
            f"{test_count:>{widths['test']}}"
            f"{class_total:>{widths['total']}}"
        )
    total_all = sum(total_by_split.values())
    print(
        f"{'TOTAL':<{widths['label']}}"
        f"{total_by_split['train']:>{widths['train']}}"
        f"{total_by_split['valid']:>{widths['valid']}}"
        f"{total_by_split['test']:>{widths['test']}}"
        f"{total_all:>{widths['total']}}"
    )
    print()


def report_duplicate_stems(split_inventory: dict[str, SplitInventory]) -> None:
    print("Duplicate name-base validation failed.")
    for split in SPLITS:
        inventory = split_inventory[split]
        if inventory.duplicate_images:
            print(f"{split} duplicate image stems:")
            for stem, members in inventory.duplicate_images.items():
                print(f"- {stem}")
                for member in members:
                    print(f"  {member}")
        if inventory.duplicate_labels:
            print(f"{split} duplicate label stems:")
            for stem, members in inventory.duplicate_labels.items():
                print(f"- {stem}")
                for member in members:
                    print(f"  {member}")


def create_staging_dataset(
    zip_path: Path,
    output_root: Path,
    analysis: ArchiveAnalysis,
    split_plan: SplitPlan,
) -> Path:
    stage_root = Path(tempfile.mkdtemp(prefix="plant_disease_redistribute_stage_"))
    try:
        with ZipFile(zip_path, "r") as archive:
            for split in SPLITS:
                (stage_root / split / "images").mkdir(parents=True, exist_ok=True)
                (stage_root / split / "labels").mkdir(parents=True, exist_ok=True)

            for class_id, split_map in split_plan.by_class.items():
                for split, items in split_map.items():
                    for item in items:
                        copy_zip_member(
                            archive,
                            item.image_member,
                            stage_root
                            / split
                            / "images"
                            / PurePosixPath(item.image_member).name,
                        )
                        copy_zip_member(
                            archive,
                            item.label_member,
                            stage_root
                            / split
                            / "labels"
                            / PurePosixPath(item.label_member).name,
                        )

        write_output_data_yaml(stage_root / "data.yaml")
        return stage_root
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def copy_zip_member(archive: ZipFile, member_name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member_name, "r") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=CHUNK_SIZE)


def write_output_data_yaml(data_yaml_path: Path) -> None:
    data_yaml_path.write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                "nc: 3",
                "names: ['mildew', 'rose_P01', 'rose_R02']",
                "",
            ]
        ),
        encoding="utf-8",
    )


def validate_staged_dataset(
    stage_root: Path, analysis: ArchiveAnalysis, split_plan: SplitPlan
) -> None:
    seen_images: set[str] = set()
    seen_labels: set[str] = set()
    total_images = 0
    total_labels = 0

    for split in SPLITS:
        images_dir = stage_root / split / "images"
        labels_dir = stage_root / split / "labels"
        images = sorted(
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        labels = sorted(
            path
            for path in labels_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        )
        total_images += len(images)
        total_labels += len(labels)

        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise ArchiveIntegrityError(f"Staged pairing mismatch in split '{split}'.")

        for stem in image_stems:
            if stem in seen_images:
                raise ArchiveIntegrityError("Image stem duplicated across splits.")
            if stem in seen_labels:
                raise ArchiveIntegrityError("Label stem duplicated across splits.")
            seen_images.add(stem)
            seen_labels.add(stem)

        class_distribution = Counter()
        for label_path in labels:
            class_id = read_local_label_class_id(label_path)
            class_distribution[class_id] += 1
        expected = split_plan.split_counts()[split]
        if class_distribution != expected:
            raise ArchiveIntegrityError(
                f"Class distribution mismatch in staged split '{split}'."
            )

        if not all(class_distribution[class_id] > 0 for class_id in EXPECTED_CLASS_IDS):
            raise ArchiveIntegrityError(
                f"Not all classes are present in staged split '{split}'."
            )

    if total_images != len(analysis.global_valid_pairs) or total_labels != len(
        analysis.global_valid_pairs
    ):
        raise ArchiveIntegrityError("Staged totals do not match the valid pair count.")


def read_local_label_class_id(label_path: Path) -> int:
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ArchiveIntegrityError(f"Empty label file in staged dataset: {label_path}")
    first = lines[0].split()
    if not first:
        raise ArchiveIntegrityError(
            f"Invalid label file in staged dataset: {label_path}"
        )
    class_id = int(first[0])
    if class_id not in EXPECTED_CLASS_IDS:
        raise ArchiveIntegrityError(
            f"Unexpected class id {class_id} in staged dataset."
        )
    return class_id


def stage_to_final_output(stage_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise RedistributionAbort(
            f"Output directory already exists: {output_root}. Aborting for safety."
        )
    stage_root.replace(output_root)


@dataclass(slots=True)
class FinalSplitReport:
    images: int
    labels: int
    class_counts: Counter[int]


@dataclass(slots=True)
class FinalReport:
    split_reports: dict[str, FinalSplitReport]

    @property
    def total_images(self) -> int:
        return sum(report.images for report in self.split_reports.values())

    @property
    def total_labels(self) -> int:
        return sum(report.labels for report in self.split_reports.values())


def analyze_redistributed_output(output_root: Path) -> FinalReport:
    split_reports: dict[str, FinalSplitReport] = {}
    for split in SPLITS:
        images_dir = output_root / split / "images"
        labels_dir = output_root / split / "labels"
        images = [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        labels = [
            path
            for path in labels_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        ]

        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise ArchiveIntegrityError(
                f"Pairing mismatch after redistribution in '{split}'."
            )

        class_counts = Counter()
        for label_path in labels:
            class_id = read_local_label_class_id(label_path)
            class_counts[class_id] += 1

        split_reports[split] = FinalSplitReport(
            images=len(images),
            labels=len(labels),
            class_counts=class_counts,
        )
    return FinalReport(split_reports=split_reports)


def print_final_validation(report: FinalReport) -> None:
    print("=" * 50)
    print("FINAL DATASET VALIDATION")
    print("=" * 50)
    print()
    for split in SPLITS:
        split_report = report.split_reports[split]
        print(f"{split.upper()}:")
        print(f"images: {split_report.images}")
        print(f"labels: {split_report.labels}")
        for class_id in sorted(EXPECTED_CLASS_IDS):
            print(f"class {class_id}: {split_report.class_counts[class_id]}")
        print()
    print("TOTAL:")
    print(f"images: {report.total_images}")
    print(f"labels: {report.total_labels}")
    print()
    print("PAIRING: OK")
    print("DUPLICATES BETWEEN SPLITS: NONE")
    all_splits_have_all_classes = all(
        all(
            report.split_reports[split].class_counts[class_id] > 0
            for class_id in EXPECTED_CLASS_IDS
        )
        for split in SPLITS
    )
    print(
        f"ALL CLASSES PRESENT IN ALL SPLITS: {'YES' if all_splits_have_all_classes else 'NO'}"
    )
    print()
    print("Real split percentages:")
    total_pairs = report.total_images
    for split in SPLITS:
        split_count = report.split_reports[split].images
        percentage = (split_count / total_pairs) * 100 if total_pairs else 0.0
        print(f"{split}: {split_count} ({percentage:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except ArchiveIntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    except RedistributionAbort as exc:
        raise SystemExit(str(exc)) from exc
