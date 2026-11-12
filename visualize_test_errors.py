"""Create visual summaries for TEST-set prediction errors.

This script is read-only with respect to the model checkpoint and the dataset.
It consumes the existing predictions CSV produced by the TEST evaluation step
and generates contact sheets for the full error set and each confusion type.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_CSV = PROJECT_ROOT / "outputs" / "test_predictions.csv"
TEST_IMAGES_DIR = PROJECT_ROOT / "dataset_redistributed" / "test" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_error_review"

CLASS_NAMES = ("mildew", "rose_P01", "rose_R02")
CONFUSION_PAIRS = (
    ("mildew", "rose_P01"),
    ("mildew", "rose_R02"),
    ("rose_P01", "mildew"),
    ("rose_P01", "rose_R02"),
    ("rose_R02", "mildew"),
    ("rose_R02", "rose_P01"),
)
ALL_ERRORS_OUTPUT = OUTPUT_DIR / "all_errors.jpg"
CONFUSION_OUTPUTS = {
    ("mildew", "rose_P01"): OUTPUT_DIR / "mildew_to_rose_P01.jpg",
    ("mildew", "rose_R02"): OUTPUT_DIR / "mildew_to_rose_R02.jpg",
    ("rose_P01", "mildew"): OUTPUT_DIR / "rose_P01_to_mildew.jpg",
    ("rose_P01", "rose_R02"): OUTPUT_DIR / "rose_P01_to_rose_R02.jpg",
    ("rose_R02", "mildew"): OUTPUT_DIR / "rose_R02_to_mildew.jpg",
    ("rose_R02", "rose_P01"): OUTPUT_DIR / "rose_R02_to_rose_P01.jpg",
}
EXPECTED_ERROR_COUNT = 25
GRID_COLUMNS = 5
ALL_ERRORS_ROWS = 5
CELL_WIDTH = 360
CELL_HEIGHT = 320
IMAGE_AREA_HEIGHT = 240
MOSAIC_PADDING = 16
CELL_SPACING = 10
BACKGROUND_COLOR = (248, 248, 248)
CELL_BACKGROUND_COLOR = (255, 255, 255)
TEXT_COLOR = (20, 20, 20)
ACCENT_COLOR = (90, 90, 90)


@dataclass(slots=True)
class PredictionRow:
    image: str
    true_class: int
    true_class_name: str
    predicted_class: int
    predicted_class_name: str
    confidence: float
    prob_class_0: float
    prob_class_1: float
    prob_class_2: float
    correct: bool

    @property
    def image_path(self) -> Path:
        return TEST_IMAGES_DIR / self.image


@dataclass(slots=True)
class CellData:
    image_path: Path
    title_lines: list[str]


def main() -> None:
    predictions = _load_predictions(PREDICTIONS_CSV)
    errors = [row for row in predictions if not row.correct]

    _validate_environment(predictions, errors)
    _prepare_output_dir()

    sorted_errors = sorted(errors, key=lambda row: row.confidence, reverse=True)
    _create_contact_sheet(
        ALL_ERRORS_OUTPUT,
        [
            CellData(
                image_path=row.image_path,
                title_lines=[
                    f"#{index:02d}",
                    f"REAL: {row.true_class_name}",
                    f"PRED: {row.predicted_class_name}",
                    f"CONF: {row.confidence * 100:.2f}%",
                ],
            )
            for index, row in enumerate(sorted_errors, start=1)
        ],
        columns=GRID_COLUMNS,
        rows=ALL_ERRORS_ROWS,
        title="TEST Errors",
    )

    grouped_errors = _group_errors_by_confusion(sorted_errors)
    for pair in CONFUSION_PAIRS:
        rows_for_pair = grouped_errors.get(pair, [])
        output_path = CONFUSION_OUTPUTS[pair]
        _create_contact_sheet(
            output_path,
            [
                CellData(
                    image_path=row.image_path,
                    title_lines=[
                        row.image,
                        f"REAL: {row.true_class_name}",
                        f"PRED: {row.predicted_class_name}",
                        f"CONF: {row.confidence * 100:.2f}%",
                    ],
                )
                for row in rows_for_pair
            ],
            columns=_choose_columns(len(rows_for_pair)),
            rows=_choose_rows(len(rows_for_pair), _choose_columns(len(rows_for_pair))),
            title=f"{pair[0]} -> {pair[1]}",
        )

    _print_summary(errors, grouped_errors)


def _load_predictions(csv_path: Path) -> list[PredictionRow]:
    if not csv_path.exists():
        raise SystemExit(f"Predictions CSV not found: {csv_path}")

    rows: list[PredictionRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        expected_fields = {
            "image",
            "true_class",
            "true_class_name",
            "predicted_class",
            "predicted_class_name",
            "confidence",
            "prob_class_0",
            "prob_class_1",
            "prob_class_2",
            "correct",
        }
        if set(reader.fieldnames or ()) != expected_fields:
            raise SystemExit(
                "test_predictions.csv has an unexpected schema. "
                f"Expected fields: {sorted(expected_fields)}"
            )

        for raw_row in reader:
            rows.append(
                PredictionRow(
                    image=raw_row["image"],
                    true_class=int(raw_row["true_class"]),
                    true_class_name=raw_row["true_class_name"],
                    predicted_class=int(raw_row["predicted_class"]),
                    predicted_class_name=raw_row["predicted_class_name"],
                    confidence=float(raw_row["confidence"]),
                    prob_class_0=float(raw_row["prob_class_0"]),
                    prob_class_1=float(raw_row["prob_class_1"]),
                    prob_class_2=float(raw_row["prob_class_2"]),
                    correct=_parse_bool(raw_row["correct"]),
                )
            )
    return rows


def _validate_environment(
    predictions: list[PredictionRow], errors: list[PredictionRow]
) -> None:
    if not PREDICTIONS_CSV.exists():
        raise SystemExit(f"Predictions CSV not found: {PREDICTIONS_CSV}")
    if not TEST_IMAGES_DIR.exists():
        raise SystemExit(f"TEST images directory not found: {TEST_IMAGES_DIR}")

    if len(errors) != EXPECTED_ERROR_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_ERROR_COUNT} incorrect predictions, found {len(errors)}."
        )

    for row in predictions:
        image_path = row.image_path
        if not image_path.exists():
            raise SystemExit(f"Missing TEST image: {image_path}")

    for row in errors:
        if row.true_class_name not in CLASS_NAMES:
            raise SystemExit(
                f"Unexpected true class name in CSV: {row.true_class_name}"
            )
        if row.predicted_class_name not in CLASS_NAMES:
            raise SystemExit(
                f"Unexpected predicted class name in CSV: {row.predicted_class_name}"
            )
        if row.predicted_class_name == row.true_class_name:
            raise SystemExit(
                "The error list contains a row marked as incorrect but with matching classes."
            )


def _prepare_output_dir() -> None:
    if OUTPUT_DIR.exists():
        existing_entries = list(OUTPUT_DIR.iterdir())
        if existing_entries:
            raise SystemExit(
                f"Output directory already exists and is not empty: {OUTPUT_DIR}. "
                "Remove it manually before running this script to avoid overwriting files."
            )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _group_errors_by_confusion(
    errors: list[PredictionRow],
) -> dict[tuple[str, str], list[PredictionRow]]:
    grouped: dict[tuple[str, str], list[PredictionRow]] = {}
    for pair in CONFUSION_PAIRS:
        grouped[pair] = []

    for row in errors:
        key = (row.true_class_name, row.predicted_class_name)
        grouped.setdefault(key, []).append(row)

    for pair in grouped:
        grouped[pair] = sorted(
            grouped[pair], key=lambda item: item.confidence, reverse=True
        )
    return grouped


def _create_contact_sheet(
    output_path: Path,
    cells: list[CellData],
    columns: int,
    rows: int,
    title: str,
) -> None:
    if not cells:
        raise SystemExit(f"No images available for mosaic: {output_path.name}")

    canvas_width = (
        MOSAIC_PADDING * 2 + columns * CELL_WIDTH + (columns - 1) * CELL_SPACING
    )
    canvas_height = (
        MOSAIC_PADDING * 2 + 48 + rows * CELL_HEIGHT + (rows - 1) * CELL_SPACING
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(28)
    text_font = _load_font(18)
    small_font = _load_font(16)

    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_x = (canvas_width - (title_bbox[2] - title_bbox[0])) // 2
    draw.text((title_x, MOSAIC_PADDING), title, fill=TEXT_COLOR, font=title_font)

    for index in range(columns * rows):
        cell_x = MOSAIC_PADDING + (index % columns) * (CELL_WIDTH + CELL_SPACING)
        cell_y = MOSAIC_PADDING + 48 + (index // columns) * (CELL_HEIGHT + CELL_SPACING)
        cell_box = (cell_x, cell_y, cell_x + CELL_WIDTH, cell_y + CELL_HEIGHT)
        draw.rounded_rectangle(
            cell_box,
            radius=14,
            fill=CELL_BACKGROUND_COLOR,
            outline=(220, 220, 220),
            width=2,
        )

        if index >= len(cells):
            continue

        cell = cells[index]
        image_area = (
            cell_x + 12,
            cell_y + 12,
            cell_x + CELL_WIDTH - 12,
            cell_y + IMAGE_AREA_HEIGHT,
        )
        text_area_top = cell_y + IMAGE_AREA_HEIGHT + 16
        image = _open_image(cell.image_path)
        fitted = ImageOps.contain(
            image,
            (image_area[2] - image_area[0], image_area[3] - image_area[1]),
        )
        paste_x = image_area[0] + ((image_area[2] - image_area[0]) - fitted.width) // 2
        paste_y = image_area[1] + ((image_area[3] - image_area[1]) - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))

        text_y = text_area_top
        for line_index, line in enumerate(cell.title_lines):
            font = text_font if line_index > 0 else small_font
            line_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            line_height = line_bbox[3] - line_bbox[1]
            line_x = cell_x + (CELL_WIDTH - line_width) // 2
            draw.text((line_x, text_y), line, fill=TEXT_COLOR, font=font)
            text_y += line_height + 4

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95)


def _open_image(image_path: Path) -> Image.Image:
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except OSError as exc:
        raise SystemExit(f"Unable to open image: {image_path}") from exc


def _load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _choose_columns(count: int) -> int:
    if count <= 1:
        return 1
    if count <= 4:
        return 2
    if count <= 9:
        return 3
    return 4


def _choose_rows(count: int, columns: int) -> int:
    return max(1, math.ceil(count / columns))


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def _print_summary(
    errors: list[PredictionRow],
    grouped_errors: dict[tuple[str, str], list[PredictionRow]],
) -> None:
    total_errors = len(errors)
    counts = {pair: len(grouped_errors.get(pair, [])) for pair in CONFUSION_PAIRS}

    print("==============================")
    print("TEST ERROR VISUALIZATION")
    print("==============================")
    print()
    print(f"Total errors: {total_errors}")
    print()
    for true_name, predicted_name in CONFUSION_PAIRS:
        print(f"{true_name} -> {predicted_name}: {counts[(true_name, predicted_name)]}")
    print()
    print("Files created:")
    print("outputs/test_error_review/all_errors.jpg")
    print("outputs/test_error_review/mildew_to_rose_P01.jpg")
    print("outputs/test_error_review/mildew_to_rose_R02.jpg")
    print("outputs/test_error_review/rose_P01_to_mildew.jpg")
    print("outputs/test_error_review/rose_P01_to_rose_R02.jpg")
    print("outputs/test_error_review/rose_R02_to_mildew.jpg")
    print("outputs/test_error_review/rose_R02_to_rose_P01.jpg")


if __name__ == "__main__":
    main()
