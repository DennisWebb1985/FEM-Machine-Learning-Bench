from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageSample:
    path: Path
    target: int
    class_name: str


def discover_image_samples(
    root_dir: str | Path,
    class_names: Iterable[str],
    file_extensions: Iterable[str] = (".jpg", ".jpeg", ".png"),
) -> list[ImageSample]:
    root = Path(root_dir)
    normalized_extensions = {extension.lower() for extension in file_extensions}
    samples: list[ImageSample] = []

    for target, class_name in enumerate(class_names):
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory does not exist: {class_dir}")

        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in normalized_extensions:
                samples.append(ImageSample(path=path, target=target, class_name=class_name))

    if not samples:
        raise ValueError(f"No image files found under {root}")
    return samples


def _subset_per_class(
    samples: list[ImageSample],
    max_samples_per_class: int | None,
    seed: int,
) -> list[ImageSample]:
    if max_samples_per_class is None:
        return samples

    grouped: dict[int, list[ImageSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.target, []).append(sample)

    rng = random.Random(seed)
    limited: list[ImageSample] = []
    for target in sorted(grouped):
        class_samples = grouped[target][:]
        rng.shuffle(class_samples)
        limited.extend(class_samples[:max_samples_per_class])
    return sorted(limited, key=lambda sample: (sample.target, sample.path.name))


def split_image_samples(
    samples: list[ImageSample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[ImageSample]]:
    test_ratio = 1.0 - train_ratio - val_ratio
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must leave a positive test split.")

    indices = np.arange(len(samples))
    targets = np.asarray([sample.target for sample in samples], dtype=np.int64)

    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=targets,
        random_state=seed,
    )

    temp_targets = targets[temp_indices]
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_fraction_of_temp,
        stratify=temp_targets,
        random_state=seed,
    )

    return {
        "train": [samples[index] for index in train_indices],
        "val": [samples[index] for index in val_indices],
        "test": [samples[index] for index in test_indices],
    }


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        samples: list[ImageSample],
        image_size: int,
        grayscale: bool = False,
        normalize_mean: list[float] | None = None,
        normalize_std: list[float] | None = None,
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.grayscale = grayscale
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.horizontal_flip_prob = horizontal_flip_prob

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        mode = "L" if self.grayscale else "RGB"
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert(mode)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            if self.horizontal_flip_prob > 0.0 and random.random() < self.horizontal_flip_prob:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            array = np.asarray(image, dtype=np.float32) / 255.0

        if self.grayscale:
            array = np.expand_dims(array, axis=0)
        else:
            array = np.transpose(array, (2, 0, 1))

        tensor = torch.from_numpy(array)

        if self.normalize_mean is not None and self.normalize_std is not None:
            mean = torch.tensor(self.normalize_mean, dtype=torch.float32).view(-1, 1, 1)
            std = torch.tensor(self.normalize_std, dtype=torch.float32).view(-1, 1, 1)
            tensor = (tensor - mean) / std

        return tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = self._load_image(sample.path)
        target = torch.tensor(sample.target, dtype=torch.long)
        return image, target

    def target_array(self) -> np.ndarray:
        return np.asarray([sample.target for sample in self.samples], dtype=np.int64)

    def metadata_frame(self) -> list[dict[str, Any]]:
        return [
            {
                "path": str(sample.path),
                "class_name": sample.class_name,
                "target": sample.target,
            }
            for sample in self.samples
        ]
