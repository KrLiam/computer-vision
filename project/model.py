import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, ToPILImage

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

MODEL_PATH = "model.pth"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        print("Built neural network")

    def forward(self, x):
        print("x", x)
        x = self.flatten(x)
        print("flatten", x)
        logits = self.linear_relu_stack(x)
        print("logits", logits)
        return logits


def get_dataset() -> tuple[DataLoader, DataLoader]:
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

        if batch % 100 == 0:
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
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def save_model(model: NeuralNetwork, path: str):
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")

def load_model(path: str, model: NeuralNetwork | None = None) -> NeuralNetwork:
    if model is None:
        model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def run_training():
    # Dataset
    train_dataloader, test_dataloader = get_dataset()

    # Initialize model
    model = NeuralNetwork().to(device)
    try:
        load_model(MODEL_PATH, model)
        print(f"Model '{MODEL_PATH}' loaded successfully!")
    except FileNotFoundError:
        ...

    # Optimize model parameters
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

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
    train_dataloader, test_dataloader = get_dataset()
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

def run():
    run_test()
