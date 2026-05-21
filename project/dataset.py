import glob
import os
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
    transform = ToTensor()
) -> torch.Tensor:
    converted: list[torch.Tensor] = []
    for img in imgs:
        if isinstance(img, MatLike):
            img = Image.fromarray(img)

        img = img.convert("L")
        converted.append(transform(img).squeeze(0))

    return torch.stack(converted)

def get_dataset_samples(patterns: list[str]) -> dict[str, list[tuple[int, str]]]:
    samples = defaultdict(list)

    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
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
            
    return samples

def build_dataset(
    patterns: list[str],
    output_path: str = "dataset.tar",
    first_note: int = 36, # C2
    num_notes: int = 61,
):
    print("Building dataset")

    samples = get_dataset_samples(patterns)
    
    print(f"Found {len(samples)} samples")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
    if output_path.endswith(".tar"):
        metadata = []
        with tarfile.open(output_path, "w") as tar:
            for i, (key, frames) in enumerate(samples.items()):
                name = os.path.basename(key)
                notes = tuple(sorted(name.split('_')))
                frames.sort(key=lambda x: x[0])
                
                y = [0.0] * num_notes
                for note_str in notes:
                    code = get_note_code(note_str)
                    if code is None:
                        continue
                    code -= first_note
                    if 0 <= code < num_notes:
                        y[code] = 1.0
                        
                sample_meta = {
                    "id": i,
                    "y": y,
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
        x_tensors = []
        y_tensors = []
        
        for key, frames in samples.items():
            name = os.path.basename(key)
            notes = tuple(sorted(name.split('_')))
                
            frames.sort(key=lambda x: x[0])
            
            images = [Image.open(path) for _, path in frames]
            x = frames_to_tensor(images) # Shape: (frames, height, width)
            
            y = torch.zeros(num_notes, dtype=torch.float32)
            for note_str in notes:
                code = get_note_code(note_str)
                if code is None:
                    continue

                code -= first_note
                if 0 <= code < num_notes:
                    y[code] = 1.0
                            
            x_tensors.append(x)
            y_tensors.append(y)
        
        x_shape = tuple(x_tensors[0].shape)
        y_shape = tuple(y_tensors[0].shape)

        print(f"Dataset built, X={x_shape}, Y={y_shape}")

        data = { 'x': torch.stack(x_tensors), 'y': torch.stack(y_tensors) }
        torch.save(data, output_path)
        print(f"Dataset saved to {output_path}")

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
        num_workers=4,
        pin_memory=True
    )

def dataset_info(path: str, first_note: int = 36):
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
