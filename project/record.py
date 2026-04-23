import datetime
import os
from pathlib import Path
import threading
from collections import deque

import cv2
from cv2.typing import MatLike
import mido

from project.crop import CroppingRegion

# Disable Kivy's argument parser to avoid conflicts with our own argparse
os.environ["KIVY_NO_ARGS"] = "1"
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label

from project.midi import format_note


class MidiListener:
    def __init__(self, target_port: str):
        self.target_port = target_port
        self.midi_log = []
        self._running = False
        self._thread = None

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

    def get_messages(self):
        messages = self.midi_log[:]
        self.midi_log.clear()
        return messages

    def stop(self):
        self._running = False


def video_capture(device: str) -> cv2.VideoCapture:
    if device.isdigit():
        # windows
        device = int(device)
        cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    else:
        # elsewhere
        cap = cv2.VideoCapture(device)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    return cap


def labelled_checkbox(label: str, active: bool = False):
    box = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
    box.add_widget(Label(text=label, size_hint_x=0.3))
    checkbox = CheckBox(active=active, size_hint_x=0.7)
    box.add_widget(checkbox)
    return checkbox

class RecordingApp(App):
    def __init__(self, target_port: str, video_device: str, **kwargs):
        super().__init__(**kwargs)
        self.video_device = video_device
        self.midi_listener = MidiListener(target_port)
        self.cap = None
        self.frame_buffer = deque(maxlen=3)
        self.pending_notes = []

        self.cap = video_capture(self.video_device)
        if not self.cap.isOpened():
            print(f"Camera device '{self.video_device}' is invalid.")
            exit()

        frame = self.get_frame()
        if frame is not None:
            self.frame_buffer.append(frame)

    def build(self):
        os.makedirs("frames", exist_ok=True)

        layout = BoxLayout(orientation='horizontal')

        self.camera_view = Image(size_hint=(0.7, 1.0))
        self.camera_view.bind(on_touch_down=self.on_camera_touch)
        layout.add_widget(self.camera_view)

        sidebar = BoxLayout(orientation='vertical', size_hint=(0.3, 1.0))

        w, h = 0, 0
        frame = self.frame_buffer[0]
        if frame is not None:
            h, w, _ = frame.shape
        self.cropping_region = CroppingRegion(default_w=w, default_h=h)
        sidebar.add_widget(self.cropping_region.build())

        self.flip_x_cb = labelled_checkbox("Flip X:")
        sidebar.add_widget(self.flip_x_cb.parent)

        self.flip_y_cb = labelled_checkbox("Flip Y:")
        sidebar.add_widget(self.flip_y_cb.parent)

        self.gray_cb = labelled_checkbox("Grayscale:")
        sidebar.add_widget(self.gray_cb.parent)

        self.captured_label = Label(
            text="Captured: ",
            valign='top',
            halign='left',
            padding=(10, 10)
        )
        self.captured_label.bind(size=self.captured_label.setter('text_size'))
        sidebar.add_widget(self.captured_label)
        layout.add_widget(sidebar)

        self.midi_listener.start()

        # Match Kivy refresh interval to 30 FPS.
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return layout

    def get_frame(self) -> MatLike | None:
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def on_camera_touch(self, instance, touch):
        if not instance.collide_point(*touch.pos):
            return False

        if not self.frame_buffer or self.frame_buffer[0] is None:
            return False

        ix, iy = instance.norm_image_size
        if ix == 0 or iy == 0:
            return False

        cx, cy = instance.center
        bx, by = cx - ix / 2, cy - iy / 2
        x, y = touch.pos

        if bx <= x <= bx + ix and by <= y <= by + iy:
            rel_x = (x - bx) / ix
            rel_y = ((by + iy) - y) / iy
            h, w = self.frame_buffer[0].shape[:2]
            self.cropping_region.push_point(int(rel_x * w), int(rel_y * h))
            return True
        return False

    def update(self, dt):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_buffer.append(frame)
        self._update_camera_view(frame)
        self._process_frames()
        self._process_midi()
        self._update_sidebar()

    def _transform_frame(self, frame, outline: bool = False):
        if self.gray_cb.active:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if self.flip_y_cb.active:
            frame = cv2.flip(frame, 0)

        if self.flip_x_cb.active:
            frame = cv2.flip(frame, 1)
            
        if outline:
            self.cropping_region.draw_outline(frame)
        else:
            frame = self.cropping_region.apply(frame)

        return frame


    def _update_camera_view(self, frame):
        frame = frame.copy()
        frame = self._transform_frame(frame, outline=True)

        # Convert BGR to RGB and flip vertically for Kivy Texture
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera_view.texture = texture

    def _process_frames(self):
        for idx, (c, notes) in reversed(list(enumerate(self.pending_notes))):
            if c == 0:
                del self.pending_notes[idx]
                notes_str = "_".join(notes)
                for frame_idx, b_frame in enumerate(self.frame_buffer):
                    filepath = Path("frames") / f"{notes_str}_{frame_idx}.png"

                    cropped = self._transform_frame(b_frame)
                    cv2.imwrite(filepath, cropped)
                    print(f"Saved frame {frame_idx} to {filepath}")
            else:
                self.pending_notes[idx] = (c - 1, notes)

    def _process_midi(self):
        messages = self.midi_listener.get_messages()

        if messages:
            print(len(messages))

        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-4]

        for msg in messages:
            if not hasattr(msg, "note"):
                continue

            pressed = msg.type == "note_on"
            note = msg.note
            formatted_note = format_note(note)

            if pressed:
                c = 1
                merge_tolerance = 1

                if self.pending_notes and self.pending_notes[-1][0] - c <= merge_tolerance:
                    self.pending_notes[-1][1].append(formatted_note)
                else:
                    self.pending_notes.append((c, [formatted_note]))
            else:
                print(f"[{current_time}] Released note {formatted_note}")

    def _update_sidebar(self):
        if not self.pending_notes:
            self.captured_label.text = "Captured: "
            return

        lines = ["Captured: "]
        for _, notes in self.pending_notes:
            lines.append('+'.join(notes))
        self.captured_label.text = ", ".join(lines)

    def on_stop(self):
        if self.cap:
            self.cap.release()
        self.midi_listener.stop()
        print("\nStopped listening to MIDI input.")


def run_recording(midi_name: str, video_device: str):
    available_ports = mido.get_input_names()
    target_port = next((p for p in available_ports if midi_name in p), None)

    if not target_port:
        print(f"Could not find a MIDI port containing '{midi_name}'.")
        s = ", ".join(available_ports)
        print(f"Available ports: {s}")
        return

    RecordingApp(target_port, video_device).run()
