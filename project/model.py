import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, ToPILImage

from project.dataset import get_fashion_dataset, load_dataset
from project.midi import tensor_to_notes

device = (
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
        X, y = X.to(device), y.to(device)

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

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    min_pred, max_pred = 1.0, 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
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
    print(f"Test:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, Pred Range: [{min_pred:>0.4f}, {max_pred:>0.4f}]\n")


def save_model(model: NeuralNetwork, path: str):
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")


def load_model(path: str, model: NeuralNetwork | None = None) -> NeuralNetwork:
    if model is None:
        model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def run_training(train_dataset: str, test_dataset: str, batch_size: int = 32):
    # Dataset
    train_dataloader = load_dataset(train_dataset, batch_size=batch_size)
    test_dataloader = load_dataset(test_dataset, batch_size=batch_size)
    print(f"Loaded train dataset '{train_dataset}' and test dataset '{test_dataset}' with batch size {batch_size}")

    # Initialize model
    model = NeuralNetwork().to(device)
    try:
        load_model(MODEL_PATH, model)
        print(f"Model '{MODEL_PATH}' loaded successfully!")
    except FileNotFoundError:
        ...

    # Optimize model parameters

    loss_fn = nn.BCEWithLogitsLoss() # Multi-class
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    test(test_dataloader, model, loss_fn)

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    print("Done!")

    save_model(model, MODEL_PATH)


def run_test():
    # Dataset
    train_dataloader, test_dataloader = get_fashion_dataset()
    test_data = test_dataloader.dataset
    classes: list[str] = test_data.classes

    # Initialize model
    model = load_model(MODEL_PATH)

    model.eval()
    i = random.randint(0, len(test_data) - 1)
    x, y = test_data[i][0], test_data[i][1]

    ToPILImage()(x).show()

    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
