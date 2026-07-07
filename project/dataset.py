import glob
import glob
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable
import tarfile
import json
import io

from cv2.typing import MatLike
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from project.midi import format_note, get_note_code, tensor_to_notes

EXPECTED_IMAGE_SIZE = (640, 128)
PT_DATASET_MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024


def get_image_paths(patterns: list[str]) -> list[str]:
    paths = []
    for pattern in patterns:
        normalized_pattern = os.path.normpath(pattern)
        paths.extend(
            path
            for path in glob.glob(normalized_pattern, recursive=True)
            if os.path.isfile(path)
        )
    return paths


def get_image_size_errors(
    patterns: list[str],
    expected_size: tuple[int, int] = EXPECTED_IMAGE_SIZE,
) -> tuple[int, list[tuple[str, tuple[int, int]]]]:
    paths = get_image_paths(patterns)
    errors = []

    for path in paths:
        with Image.open(path) as img:
            if img.size != expected_size:
                errors.append((path, img.size))

    return len(paths), errors

def get_fashion_dataset() -> tuple[DataLoader, DataLoader]:
    print("Downloading dataset.")

    # Download training data from open datasets
    training_data = FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    # Download test data from open datasets
    test_data = FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


def frames_to_tensor(
    imgs: Iterable[Image.Image | MatLike],
    transform = ToTensor(),
    size: tuple[int, int] = (640, 128),
) -> torch.Tensor:
    converted: list[torch.Tensor] = []
    for img in imgs:
        if isinstance(img, MatLike):
            img = Image.fromarray(img)

        img = img.convert("L")
        if img.size != size:
            img = img.resize(size)
        converted.append(transform(img).squeeze(0))

    return torch.stack(converted)

def get_dataset_samples(
    patterns: list[str],
    cap_none: bool = False,
    first_note: int = 36, # C2
    num_notes: int = 61,
    seed: int = 42,
) -> dict[str, list[tuple[int, str]]]:
    samples = defaultdict(list)

    for pattern in patterns:
        normalized_pattern = os.path.normpath(pattern)
        for path in glob.glob(normalized_pattern, recursive=True):
            basename = os.path.basename(path)
            name, _ = os.path.splitext(basename)
            parts = name.split('_')
            
            if len(parts) < 2:
                continue

            *notes, frame_str = parts   
            key = f"{os.path.dirname(path)}/{'_'.join(notes)}"
            
            try:
                frame_idx = int(frame_str)
            except ValueError:
                continue
                
            samples[key].append((frame_idx, path))

    for key in list(samples.keys()):
        frames = samples[key]
        if len(frames) != 3:
            del samples[key]

    if cap_none:
        samples = cap_none_samples(samples, first_note, num_notes, seed)
            
    return samples


def cap_none_samples(
    samples: dict[str, list[tuple[int, str]]],
    first_note: int = 36, # C2
    num_notes: int = 61,
    seed: int = 42,
) -> dict[str, list[tuple[int, str]]]:
    note_counts = defaultdict(int)
    none_keys = []

    for key in samples:
        y = _sample_target(_sample_notes(key), first_note, num_notes)
        if any(y):
            for idx, active in enumerate(y):
                if active:
                    note_counts[idx] += 1
        else:
            none_keys.append(key)

    if not note_counts or not none_keys:
        return dict(samples)

    max_note_samples = max(note_counts.values())
    random.Random(seed).shuffle(none_keys)
    kept_none_keys = set(none_keys[:max_note_samples])
    capped = {
        key: frames
        for key, frames in samples.items()
        if key not in none_keys or key in kept_none_keys
    }

    removed = len(none_keys) - len(kept_none_keys)
    print(
        f"Capped none samples from {len(none_keys)} to {len(kept_none_keys)} "
        f"(removed {removed})"
    )
    return capped


def split_dataset_samples(
    samples: dict[str, list[tuple[int, str]]],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, list[tuple[int, str]]], dict[str, list[tuple[int, str]]]]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    keys = list(samples.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    test_count = round(len(keys) * test_ratio)
    if len(keys) > 1:
        test_count = min(max(1, test_count), len(keys) - 1)

    labels_by_key = {key: _sample_split_labels(key) for key in keys}
    label_totals = defaultdict(int)
    for labels in labels_by_key.values():
        for label in labels:
            label_totals[label] += 1

    desired_test_totals = {
        label: total * test_ratio
        for label, total in label_totals.items()
    }
    test_label_totals = defaultdict(int)
    remaining_keys = keys.copy()
    test_keys = set()

    while remaining_keys and len(test_keys) < test_count:
        best_key = max(
            remaining_keys,
            key=lambda key: _split_balance_score(
                labels_by_key[key],
                label_totals,
                desired_test_totals,
                test_label_totals,
            ),
        )
        test_keys.add(best_key)
        remaining_keys.remove(best_key)

        for label in labels_by_key[best_key]:
            test_label_totals[label] += 1

    train_samples = {key: samples[key] for key in keys if key not in test_keys}
    test_samples = {key: samples[key] for key in keys if key in test_keys}
    return train_samples, test_samples


def _sample_split_labels(key: str) -> tuple[str, ...]:
    labels = tuple(note for note in _sample_notes(key) if get_note_code(note) is not None)
    return labels or ("__none__",)


def _split_balance_score(
    labels: tuple[str, ...],
    label_totals: dict[str, int],
    desired_test_totals: dict[str, float],
    test_label_totals: dict[str, int],
) -> tuple[float, float]:
    score = 0.0
    rarity = 0.0

    for label in labels:
        total = label_totals[label]
        desired = desired_test_totals[label]
        current = test_label_totals[label]

        before_error = abs(current - desired)
        after_error = abs(current + 1 - desired)
        score += (before_error - after_error) / total
        rarity += 1 / total

    return score, rarity


def _sample_notes(key: str) -> tuple[str, ...]:
    name = os.path.basename(key)
    return tuple(sorted(name.split('_')))


def _sample_target(notes: tuple[str, ...], first_note: int, num_notes: int) -> list[float]:
    y = [0.0] * num_notes
    for note_str in notes:
        code = get_note_code(note_str)
        if code is None:
            continue
        code -= first_note
        if 0 <= code < num_notes:
            y[code] = 1.0
    return y


def build_dataset_from_samples(
    samples: dict[str, list[tuple[int, str]]],
    output_path: str = "dataset.tar",
    first_note: int = 36, # C2
    num_notes: int = 61,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if len(samples) == 0:
        raise ValueError("No valid samples found to build the dataset")

    if output_path.endswith(".tar"):
        metadata = []
        with tarfile.open(output_path, "w") as tar:
            for i, (key, frames) in enumerate(samples.items()):
                notes = _sample_notes(key)
                frames.sort(key=lambda x: x[0])

                sample_meta = {
                    "id": i,
                    "y": _sample_target(notes, first_note, num_notes),
                    "frames": []
                }

                for f_idx, (_, path) in enumerate(frames):
                    arcname = f"sample_{i}_{f_idx}{os.path.splitext(path)[1]}"
                    tar.add(path, arcname=arcname)
                    sample_meta["frames"].append(arcname)

                metadata.append(sample_meta)

            meta_bytes = json.dumps(metadata).encode('utf-8')
            meta_info = tarfile.TarInfo("metadata.json")
            meta_info.size = len(meta_bytes)
            tar.addfile(meta_info, io.BytesIO(meta_bytes))

        print(f"Dataset saved to {output_path} with {len(metadata)} samples")
    else:
        estimated_bytes = len(samples) * 3 * EXPECTED_IMAGE_SIZE[0] * EXPECTED_IMAGE_SIZE[1] * 4
        if estimated_bytes > PT_DATASET_MEMORY_LIMIT_BYTES:
            estimated_gb = estimated_bytes / 1024**3
            limit_gb = PT_DATASET_MEMORY_LIMIT_BYTES / 1024**3
            raise MemoryError(
                f"Refusing to build '{output_path}' as a .pt dataset because it would "
                f"allocate about {estimated_gb:.1f} GB in memory (limit: {limit_gb:.1f} GB). "
                "Use a .tar output path instead, e.g. datasets/6_train.tar."
            )

        x_tensors = []
        y_tensors = []

        for key, frames in samples.items():
            notes = _sample_notes(key)
            frames.sort(key=lambda x: x[0])

            images = [Image.open(path) for _, path in frames]
            x = frames_to_tensor(images) # Shape: (frames, height, width)
            y = torch.tensor(_sample_target(notes, first_note, num_notes), dtype=torch.float32)

            x_tensors.append(x)
            y_tensors.append(y)

        x_shape = tuple(x_tensors[0].shape)
        y_shape = tuple(y_tensors[0].shape)

        print(f"Dataset built, X={x_shape}, Y={y_shape}")

        data = { 'x': torch.stack(x_tensors), 'y': torch.stack(y_tensors) }
        torch.save(data, output_path)
        print(f"Dataset saved to {output_path}")


def build_dataset(
    patterns: list[str],
    output_path: str = "dataset.tar",
    first_note: int = 36, # C2
    num_notes: int = 61,
    cap_none: bool = False,
    seed: int = 42,
):
    print("Building dataset")

    samples = get_dataset_samples(
        patterns,
        cap_none=cap_none,
        first_note=first_note,
        num_notes=num_notes,
        seed=seed,
    )
    
    print(f"Found {len(samples)} samples")
    build_dataset_from_samples(samples, output_path, first_note, num_notes)


def build_train_test_datasets(
    patterns: list[str],
    train_output_path: str = "datasets/6_train.tar",
    test_output_path: str = "datasets/6_test.tar",
    test_ratio: float = 0.2,
    seed: int = 42,
    first_note: int = 36, # C2
    num_notes: int = 61,
    cap_none: bool = False,
):
    print("Building train/test datasets")

    samples = get_dataset_samples(
        patterns,
        cap_none=cap_none,
        first_note=first_note,
        num_notes=num_notes,
        seed=seed,
    )
    train_samples, test_samples = split_dataset_samples(samples, test_ratio, seed)

    print(
        f"Found {len(samples)} samples, "
        f"using {len(train_samples)} for training and {len(test_samples)} for testing"
    )

    build_dataset_from_samples(train_samples, train_output_path, first_note, num_notes)
    build_dataset_from_samples(test_samples, test_output_path, first_note, num_notes)

class TarDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        with tarfile.open(path, "r") as tar:
            meta_file = tar.extractfile("metadata.json")
            self.metadata = json.load(meta_file)
        self.local_tar = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.local_tar is None:
            self.local_tar = tarfile.open(self.path, "r")
            
        item = self.metadata[idx]
        images = []
        for fname in item['frames']:
            f = self.local_tar.extractfile(fname)
            img = Image.open(io.BytesIO(f.read()))
            images.append(img)
        
        x = frames_to_tensor(images)
        y = torch.tensor(item['y'], dtype=torch.float32)
        return x, y

def load_dataset(path: str = "dataset.tar", batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    if path.endswith(".tar"):
        dataset = TarDataset(path)
    else:
        data = torch.load(path, weights_only=True)
        dataset = TensorDataset(data['x'], data['y'])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

def dataset_info(path: str, first_note: int = 36, sort: bool = False):
    loader = load_dataset(path, shuffle=False)
    dataset = loader.dataset

    print(f"Samples: {len(dataset)}")
    
    if hasattr(dataset, "metadata"):
        sample_x, sample_y = dataset[0]
        print(f"X Shape: {tuple(sample_x.shape)}")
        print(f"Y Shape: {tuple(sample_y.shape)}")
        print("\nDistribution")

        distribution = defaultdict(int)
        for item in dataset.metadata:
            y = torch.tensor(item['y'])
            for note in tensor_to_notes(y, first_note):
                distribution[note] += 1
        
        if sort:
            distribution = dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
            for note_code, count in distribution.items():
                print(f"{format_note(note_code)}: {count}")
        else:
            for i in range(len(sample_y)):
                note_code = i + first_note
                print(f"{format_note(note_code)}: {distribution[note_code]}")
    else:
        x_tensors, y_tensors = dataset.tensors
        print(f"X Shape: {tuple(x_tensors[0].shape)}")
        print(f"Y Shape: {tuple(y_tensors[0].shape)}")
        print("\nDistribution")

        distribution = defaultdict(int)
        for y in y_tensors:
            for note in tensor_to_notes(y, first_note):
                distribution[note] += 1

        for i in range(y_tensors.shape[1]):
            note_code = i + first_note
            print(f"{format_note(note_code)}: {distribution[note_code]}")


def convert_dataset(path1: str, path2: str):
    if path1.endswith(".tar") and path2.endswith(".pt"):
        print(f"Converting {path1} to {path2}...")
        dataset = TarDataset(path1)
        
        x_tensors = []
        y_tensors = []
        
        for i in range(len(dataset)):
            x, y = dataset[i]
            x_tensors.append(x)
            y_tensors.append(y)
            
        data = { 'x': torch.stack(x_tensors), 'y': torch.stack(y_tensors) }
        torch.save(data, path2)
        print(f"Dataset saved to {path2}")
    elif path1.endswith(".pt") and path2.endswith(".tar"):
        print(f"Converting {path1} to {path2}...")
        data = torch.load(path1, weights_only=True)
        x_tensors = data['x']
        y_tensors = data['y']
        
        Path(path2).parent.mkdir(parents=True, exist_ok=True)
        metadata = []
        
        with tarfile.open(path2, "w") as tar:
            for i in range(len(x_tensors)):
                x = x_tensors[i]
                y = y_tensors[i]
                
                sample_meta = {
                    "id": i,
                    "y": y.tolist(),
                    "frames": []
                }
                
                for f_idx in range(x.shape[0]):
                    frame_tensor = x[f_idx]
                    img = Image.fromarray((frame_tensor.numpy() * 255).astype('uint8'), mode='L')
                    
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    arcname = f"sample_{i}_{f_idx}.png"
                    img_info = tarfile.TarInfo(arcname)
                    img_info.size = len(img_bytes)
                    tar.addfile(img_info, io.BytesIO(img_bytes))
                    
                    sample_meta["frames"].append(arcname)
                    
                metadata.append(sample_meta)
                
            meta_bytes = json.dumps(metadata).encode('utf-8')
            meta_info = tarfile.TarInfo("metadata.json")
            meta_info.size = len(meta_bytes)
            tar.addfile(meta_info, io.BytesIO(meta_bytes))
        print(f"Dataset saved to {path2}")
    else:
        print("Only .tar <-> .pt conversion is currently supported.")
