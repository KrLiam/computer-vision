
# Visão Computacional

## Setup

Initialize a virtual environment.

```bash
python3 -m venv .venv && source .venv/bin/activate
```

Install PyTorch based on your compute platform.

```bash
# cuda 13.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# cuda 12.8
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# cuda 12.6
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# no cuda, cpu
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Finally, install the remaining of the dependencies.

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 -m project
```
