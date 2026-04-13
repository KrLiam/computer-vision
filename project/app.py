
import mido
import datetime
import cv2
import os
import threading
from collections import deque

VIDEO_INPUT = '/dev/video2'
MIDI_INPUT = '20:0'

def format_note(note: int):
    n = (note // 12) - 1
    match note % 12:
        case 0: return f"C{n}"
        case 1: return f"C#{n}"
        case 2: return f"D{n}"
        case 3: return f"D#{n}"
        case 4: return f"E{n}"
        case 5: return f"F{n}"
        case 6: return f"F#{n}"
        case 7: return f"G{n}"
        case 8: return f"G#{n}"
        case 9: return f"A{n}"
        case 10: return f"A#{n}"
        case 11: return f"B{n}"

def run():
    available_ports = mido.get_input_names()
    
    target_port = next((p for p in available_ports if MIDI_INPUT in p), None)
    
    if not target_port:
        print(f"Could not find a MIDI port containing '{MIDI_INPUT}'.")
        print(f"Available ports: {available_ports}")
        return

    cap = cv2.VideoCapture(VIDEO_INPUT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    os.makedirs('frames', exist_ok=True)
    frame_buffer = deque(maxlen=3)

    midi_log = []
    pending_saves = []

    def midi_loop():
        with mido.open_input(target_port) as inport:
            for msg in inport:
                midi_log.append(msg)

    midi_thread = threading.Thread(target=midi_loop, daemon=True)
    midi_thread.start()

    print(f"Listening for MIDI input on: {target_port}...")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_buffer.append(frame)

            current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-4]
            
            for idx, (c, notes) in reversed(list(enumerate(pending_saves))):
                if c == 0:
                    del pending_saves[idx]
                    notes_str = "_".join(notes)
                    for frame_idx, b_frame in enumerate(frame_buffer):
                        filepath = f"frames/{notes_str}_{frame_idx}.png"
                        cv2.imwrite(filepath, b_frame)
                        print(f"Saved frame {frame_idx} to {filepath}")
                else:
                    pending_saves[idx] = (c - 1, notes)

            if midi_log:
                print(len(midi_log))

            while midi_log:
                msg = midi_log.pop(0)
                if not hasattr(msg, 'note'):
                    continue

                pressed = msg.type == "note_on"
                note = msg.note
                formatted_note = format_note(note)

                if pressed:
                    c = 1
                    merge_tolerance = 1

                    if pending_saves and pending_saves[-1][0] - c <= merge_tolerance:
                        pending_saves[-1][1].append(formatted_note)
                    else:
                        pending_saves.append((c, [formatted_note]))
                else:
                    print(f"[{current_time}] Released note {formatted_note}")
    except KeyboardInterrupt:
        print("\nStopped listening to MIDI input.")
    finally:
        cap.release()
