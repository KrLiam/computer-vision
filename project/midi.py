
from typing import Iterable

import torch


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