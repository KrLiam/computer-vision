
# Visão Computacional

## Setup

Initialize a virtual environment (Linux).

```bash
python3 -m venv .venv && source .venv/bin/activate
```

Initialize a virtual environment (Windows).

```bash
python3 -m venv .venv && source .venv/Scripts/activate
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

```bash
python -m project dataset record --midi-name "CASIO" --camera "0"
```

```bash
python -m project area --input "frames\right_hand\1\1\C7_0.png"
```
-----

## Frames

Filename structure:


{left_hand | right_hand}/{num_pressed_keys}/{fingers_index[list[1-5]]}/{Note}{Octave[1-5]}_{frame[0-3]}.png

