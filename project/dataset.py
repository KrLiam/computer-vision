import glob
import os
from collections import defaultdict
from typing import Iterable

from cv2.typing import MatLike
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
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


def build_dataset(
    patterns: list[str],
    output_path: str = "dataset.pt",
    first_note: int = 36, # C2
    num_notes: int = 61,
):
    print("Building dataset")

    # map note tuples to a list of (frame, path)
    samples = defaultdict(list)

    for pattern in patterns:
        for path in glob.glob(pattern):
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
    
    print(f"Found {len(samples)} samples")
                
    x_tensors = []
    y_tensors = []
    
    for key, frames in samples.items():
        name = os.path.basename(key)
        notes = tuple(sorted(name.split('_')))

        if len(frames) != 3:
            print(f"{notes} has {len(frames)} frames, skipping")
            continue
            
        frames.sort(key=lambda x: x[0])
        
        images = [Image.open(path) for _, path in frames]
        x = frames_to_tensor(images) # Shape: (frames, height, width)
        # x = x.to(torch.float8_e4m3fn)
        
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

def load_dataset(path: str = "dataset.pt", batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    data = torch.load(path)
    dataset = TensorDataset(data['x'], data['y'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def dataset_info(path: str, first_note: int = 36):
    loader = load_dataset(path)
    dataset = loader.dataset
    x_tensors, y_tensors = dataset.tensors


    print(f"Samples: {len(dataset)}")
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

