from pathlib import Path
import os
import random
import shutil
import traceback

import torch
from torch import nn
from torch.utils.data import DataLoader

from project.dataset import get_fashion_dataset, load_dataset
from project.midi import tensor_to_notes

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
# print(f"Using {device} device")

MODEL_PATH = "model.pth"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.features = nn.Sequential(
            # Shape: (B, 3, 128, 640)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            # Shape: (B, 16, 128, 640)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # Shape: (B, 16, 64, 320)
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # Shape: (B, 32, 64, 320)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Shape: (B, 32, 32, 160)
        )
        
        self.classifier = nn.Sequential(
            # Shape: (B, 32, 32, 160)
            nn.Flatten(),
            # Shape: (B, 32 * 32 * 160)
            nn.Linear(32 * 32 * 160, 512),
            # Shape: (B, 512)
            nn.ReLU(),
            nn.Linear(512, 61) # 61 output nodes for keys
            # Shape: (B, 61)
        )

        print("Built neural network")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    """
    Tests the model against the test dataset.
    """

    print("Testing model...")

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    min_pred, max_pred = 1.0, 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            # Apply sigmoid to convert logits to probabilities [0.0, 1.0]
            pred_prob = torch.sigmoid(pred)

            max_pred = max(max_pred, pred_prob.max().item())
            min_pred = min(min_pred, pred_prob.min().item())

            for y_row, pred_row in zip(y, pred_prob):
                y_notes = tensor_to_notes(y_row)
                pred_notes = tensor_to_notes(pred_row)
                if pred_notes:
                    print(f"{y_notes} -> {pred_notes}")

            test_loss += loss_fn(pred, y).item()
            correct += ((pred_prob > 0.5) == (y > 0.5)).all(dim=1).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, Pred Range: [{min_pred:>0.4f}, {max_pred:>0.4f}]\n")

    return correct


def save_model(model: NeuralNetwork, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(model.state_dict(), tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                ...
        raise
    print(f"Saved PyTorch Model State to {path}")


def checkpoint_dir(model_path: str) -> Path:
    path = Path(model_path)
    return path.with_name(f"{path.stem}_backups")


def checkpoint_path(model_path: str, name: str) -> str:
    return str(checkpoint_dir(model_path) / f"{name}.pth")


def save_checkpoint(model: NeuralNetwork, model_path: str, name: str) -> bool:
    try:
        save_model(model, checkpoint_path(model_path, name))
        cleanup_old_checkpoints(model_path)
        return True
    except Exception as error:
        print(f"Warning: checkpoint save failed: {error}")
        return False


def cleanup_old_checkpoints(model_path: str, keep: int = 3):
    path = checkpoint_dir(model_path)
    if not path.exists():
        return

    checkpoints = sorted(
        path.glob("*.pth"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for old_checkpoint in checkpoints[keep:]:
        try:
            old_checkpoint.unlink()
        except OSError:
            ...


def cleanup_checkpoints(model_path: str):
    path = checkpoint_dir(model_path)
    if path.exists():
        shutil.rmtree(path)
        print(f"Removed training backups from {path}")


def save_training_error_log(model_path: str, error: BaseException):
    path = checkpoint_dir(model_path)
    path.mkdir(parents=True, exist_ok=True)

    log_path = path / "error.log"
    log_path.write_text(
        "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        encoding="utf-8",
    )
    print(f"Saved training error log to {log_path}")


def load_model(path: str, model: NeuralNetwork | None = None) -> NeuralNetwork:
    if model is None:
        model = NeuralNetwork().to(DEVICE)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device(DEVICE)))
    return model


def calculate_weights(dataloader: DataLoader) -> torch.Tensor:
    # inicialize tensor 1x61
    N_pos = torch.zeros(61)
    N_total = 0
    for _, y in dataloader:
        N_total += y.size(0)
        # y possui dimensão batch x 61
        N_pos += y.sum(dim=0)

    N_neg = N_total - N_pos
    weights = N_neg / N_pos
    return weights.to(DEVICE)


def run_training(
    train_dataset: str,
    test_dataset: str,
    batch_size: int = 32,
    test_frequency: float = 1.0,
    epochs: int = 20,
    model_path: str = MODEL_PATH,
    target_accuracy: float = 1.0,
    start_from_scratch: bool = False,
):
    # Dataset
    train_dataloader = load_dataset(train_dataset, batch_size=batch_size)
    test_dataloader = load_dataset(test_dataset, batch_size=batch_size)
    print(f"Loaded train dataset '{train_dataset}' and test dataset '{test_dataset}' with batch size {batch_size}")

    # Initialize model
    model = NeuralNetwork().to(DEVICE)
    if start_from_scratch:
        print("Starting from scratch, ignoring any existing model file.")
    else:
        try:
            load_model(model_path, model)
            print(f"Model '{model_path}' loaded successfully!")
        except FileNotFoundError:
            ...

    # Optimize model parameters
    print("Calculating class weights...")
    pos_weight = calculate_weights(train_dataloader)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Multi-class
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Finished initializing model, loss function and optimizer.")

    try:
        accuracy = test(test_dataloader, model, loss_fn)
        if accuracy >= target_accuracy:
            print(f"Model already reached target accuracy ({target_accuracy*100:>.1f}%), exiting.")
            save_model(model, model_path)
            cleanup_checkpoints(model_path)
            return

        test_i = max(1.0, round(1 / test_frequency))
        for t in range(epochs):
            epoch = t + 1
            print(f"Epoch {epoch}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            if test_i > 0 and epoch % test_i == 0:
                accuracy = test(test_dataloader, model, loss_fn)
                save_checkpoint(model, model_path, f"accuracy_{accuracy*100:05.2f}_epoch_{epoch}".replace(".", "_"))
                if accuracy >= target_accuracy:
                    print(f"Model reached target accuracy ({target_accuracy*100:>.1f}%), stopping training.")
                    save_model(model, model_path)
                    cleanup_checkpoints(model_path)
                    return

        print("Done!")
        save_model(model, model_path)
        cleanup_checkpoints(model_path)
    except Exception as error:
        save_training_error_log(model_path, error)
        try:
            save_checkpoint(model, model_path, "failed_latest")
            print(f"Training failed. Backups kept in {checkpoint_dir(model_path)}")
        except Exception as checkpoint_error:
            print(f"Training failed and checkpoint save failed: {checkpoint_error}")
        raise


def run_test(
    test_dataset: str,
    batch_size: int = 32,
    model_path: str = MODEL_PATH,
):
    # Dataset
    test_dataloader = load_dataset(test_dataset, batch_size=batch_size)
    print(f"Loaded test dataset '{test_dataset}' with batch size {batch_size}")

    # Initialize model
    model = NeuralNetwork().to(DEVICE)
    try:
        load_model(model_path, model)
        print(f"Model '{model_path}' loaded successfully!")
    except FileNotFoundError:
        ...

    #pos_weight = calculate_weights(train_dataloader)
    loss_fn = nn.BCEWithLogitsLoss(
        #pos_weight=pos_weight
    ) # Multi-class

    test(test_dataloader, model, loss_fn)

