
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
# cuda 118
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
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

## Train

python -m project dataset record --midi-name "CASIO" --camera "0"

python -m project dataset build --images "frames/**/*.png" --output teste.pt

python -m project train --train-dataset teste.pt --test-dataset teste.pt

python -m project crop --path frames/**/*.png

 python -m project vision --model model.pth --camera "0"

## Frames

Filename structure:

# Verify if all dataset images are 640x128
python -c "import glob, PIL.Image; [print(f) for f in glob.glob('**/*.png', recursive=True) if PIL.Image.open(f).size != (640, 128)]"

# Remove those images
python -c "import glob, os, PIL.Image; [os.remove(f) for f in glob.glob('**/*.png', recursive=True) if PIL.Image.open(f).size != (640, 128)]"

{left_hand | right_hand}/{num_pressed_keys}/{fingers_index[list[1-5]]}/{Note}{Octave[1-5]}_{frame[0-3]}.png

## Presets

Presets are recorded from the recording screen with `Start preset recording`, saved
with `End recording`, and can be abandoned with `Discard preset`. They are stored in
the `presets` folder as looping videos. Their path follows the same structure as
frames, but the final number is the preset duration in seconds instead of a frame
index:

```text
presets/{left_hand | right_hand}/{num_pressed_keys}/{fingers_index[list[1-5]]}/{Note}{Octave[1-5]}_{duration_seconds}.mp4
```

On the testing screen, use the preset dropdown to switch between the live camera,
individual saved preset videos, or `All presets`. The player controls can pause the
selected preset or skip to the next preset.

