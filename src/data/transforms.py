"""Image transforms for the dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image
from torchvision import transforms as T


Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class DetectionTransform:
    """Apply image-only transforms and adjust YOLO boxes to the final size."""

    image_transforms: T.Compose

    def __call__(self, image: Image.Image, boxes: Tensor, classes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        transformed_image = self.image_transforms(image)
        if not isinstance(transformed_image, torch.Tensor):
            transformed_image = T.ToTensor()(transformed_image)

        if boxes.numel() == 0:
            empty_boxes = boxes.new_zeros((0, 4), dtype=torch.float32)
            empty_classes = classes.new_zeros((0,), dtype=torch.long)
            return transformed_image, empty_boxes, empty_classes

        image_height, image_width = transformed_image.shape[-2], transformed_image.shape[-1]
        scaled_boxes = _yolo_to_xyxy(boxes, image_width=image_width, image_height=image_height)
        return transformed_image, scaled_boxes, classes


def build_train_transforms(image_size: int) -> DetectionTransform:
    """Build transforms used by the training split."""

    return DetectionTransform(
        image_transforms=T.Compose(
            [
                T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03)], p=0.5),
                T.Resize((image_size, image_size), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    )


def build_validation_transforms(image_size: int) -> DetectionTransform:
    """Build transforms used by the validation split."""

    return DetectionTransform(
        image_transforms=T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    )


def build_test_transforms(image_size: int) -> DetectionTransform:
    """Build transforms used by the test split."""

    return DetectionTransform(
        image_transforms=T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    )


def _yolo_to_xyxy(boxes: Tensor, image_width: int, image_height: int) -> Tensor:
    """Convert normalized YOLO boxes to absolute xyxy coordinates."""

    scaled = boxes.to(dtype=torch.float32)
    scale_factors = torch.tensor(
        [image_width, image_height, image_width, image_height],
        dtype=torch.float32,
        device=scaled.device,
    )
    scaled = scaled * scale_factors

    x_center, y_center, width, height = scaled.unbind(dim=-1)
    x_min = x_center - width / 2.0
    y_min = y_center - height / 2.0
    x_max = x_center + width / 2.0
    y_max = y_center + height / 2.0
    return torch.stack((x_min, y_min, x_max, y_max), dim=-1)