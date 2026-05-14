
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import torch
import threading
import mido


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_CODES = {note: idx for idx, note in enumerate(NOTES)}

def format_note(note: int):
    n = (note // 12) - 1
    letter = NOTES[note % 12]
    return f"{letter}{n}"

def get_note_code(note_str: str) -> int | None:
    """Reverse of format_note, gets the MIDI note code from a string"""
    if not note_str:
        return None
        
    if len(note_str) >= 2 and note_str[1] == '#':
        note_name = note_str[:2]
        octave_str = note_str[2:]
    else:
        note_name = note_str[0]
        octave_str = note_str[1:]
        
    try:
        n = int(octave_str)
    except ValueError:
        return None
        
    
    if note_name not in NOTE_CODES:
        return None
        
    return (n + 1) * 12 + NOTE_CODES[note_name]


def tensor_to_notes(y: Iterable[float], first_note: int = 36) -> list[int]:
    return [i + first_note for i, val in enumerate(y) if val > 0.5]

def notes_to_tensor(notes: Iterable[int], num_notes: int = 61) -> torch.Tensor:
    y = torch.zeros(num_notes, dtype=torch.float32)
    
    for note in notes:
        y[note] = 1.0
    
    return y


def guess_key_positions(area: tuple[int, int], first_key: int = 36, num_keys: int = 61) -> dict[int, tuple[int, int, int, int]]:
    width, height = area
    
    num_major_keys = 0
    for i in range(num_keys):
        code = first_key + i
        if (code % 12) in [0, 2, 4, 5, 7, 9, 11]:
            num_major_keys += 1

    if num_major_keys == 0:
        return {}

    w_major = width / num_major_keys
    h_major = height
    w_minor = w_major * 0.6
    h_minor = height * 0.6

    positions = {}
    major_idx = 0

    for i in range(num_keys):
        code = first_key + i
        is_major = (code % 12) in [0, 2, 4, 5, 7, 9, 11]
        
        if is_major:
            x = major_idx * w_major
            y = 0
            w = w_major
            h = h_major
            positions[code] = (int(x), int(y), int(w), int(h))
            major_idx += 1
        else:
            x = (major_idx - 1) * w_major + 0.75*w_major
            y = 0
            w = w_minor
            h = h_minor
            positions[code] = (int(x), int(y), int(w), int(h))
            
    return positions

@dataclass(frozen=True)
class Note:
    code: int

    @property
    def name(self) -> str:
        return format_note(self.code)


class MidiListener:
    _pressed: deque[set[Note]]

    def __init__(self, target_port: str):
        self.target_port = target_port
        self.midi_log = []
        self._running = False
        self._thread = None
        self._pressed = deque(maxlen=30)
        self._pressed.appendleft(set())

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._midi_loop, daemon=True)
        self._thread.start()
        print(f"Listening for MIDI input on: {self.target_port}...")

    def _midi_loop(self):
        with mido.open_input(self.target_port) as inport:
            for msg in inport:
                if not self._running:
                    break
                self.midi_log.append(msg)

    def update(self):
        pressed_notes = self._pressed[0].copy()

        for msg in self.midi_log:
            pressed = msg.type == "note_on"
            note = msg.note

            if pressed:
                pressed_notes.add(Note(note))
            else:
                pressed_notes.discard(Note(note))

        self._pressed.appendleft(pressed_notes)
        self.midi_log.clear()

    def pressed(self, t: int = 0) -> Iterable[Note]:
        if t >= len(self._pressed):
            return iter(set())
        return iter(self._pressed[t])

    def just_pressed(self, t: int = 0) -> Iterable[Note]:
        if t + 1 >= len(self._pressed):
            return set()
        return self._pressed[t] - self._pressed[t + 1]
    
    def just_released(self, t: int = 0) -> Iterable[Note]:
        if t + 1 >= len(self._pressed):
            return set()
        return self._pressed[t + 1] - self._pressed[t]

    def stop(self):
        self._running = False
