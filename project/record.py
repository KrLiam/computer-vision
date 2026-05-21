from dataclasses import dataclass, field
import datetime
import json
import os
import subprocess
import sys
import logging
from pathlib import Path
import threading
from collections import deque

import cv2
from cv2.typing import MatLike
import mido

from project.crop import CroppingRegion, labelled_checkbox, text_input
from project.dataset import build_dataset
from project.model import run_training

# Disable Kivy's argument parser to avoid conflicts with our own argparse
os.environ["KIVY_NO_ARGS"] = "1"
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton

from project.midi import MidiListener, Note
from project.image_view import ImageView

logging.getLogger('PIL').setLevel(logging.WARNING)

LAST_SKEW_CONFIG_PATH = Path("frames") / "last_skew.json"


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

def labelled_dropdown(
    label: str,
    options: list[str],
    default: str,
    changed=None,
    max_height: int | None = None,
):
    box = BoxLayout(orientation='horizontal', size_hint_y=None, height=36)
    box.add_widget(Label(text=label, size_hint_x=0.45))

    button = Button(text=default, size_hint_x=0.55)
    dropdown = DropDown(max_height=max_height)

    for option in options:
        item = Button(text=option, size_hint_y=None, height=36)
        item.bind(on_release=lambda btn: dropdown.select(btn.text))
        dropdown.add_widget(item)

    button.dropdown = dropdown
    button.bind(on_release=lambda btn: btn.dropdown.open(btn))

    def on_select(_, value):
        button.text = value
        if changed is not None:
            changed(value)

    dropdown.bind(on_select=on_select)
    box.add_widget(button)
    return button


def finger_index_options() -> list[str]:
    options = []
    for mask in range(1, 1 << 5):
        fingers = [str(idx + 1) for idx in range(5) if mask & (1 << idx)]
        options.append("-".join(fingers))
    return options


class DatasetMenu(BoxLayout):
    def __init__(self, title: str, default_new_path: str, **kwargs):
        super().__init__(orientation='vertical', size_hint_y=None, height=140, **kwargs)
        self.add_widget(Label(text=title, size_hint_y=None, height=30, bold=True))

        self.toggle_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
        self.btn_new = ToggleButton(text="New", group=f"ds_{title}", state="down")
        self.btn_existing = ToggleButton(text="Existing", group=f"ds_{title}")
        self.btn_new.bind(on_press=self.on_mode_change)
        self.btn_existing.bind(on_press=self.on_mode_change)
        self.toggle_layout.add_widget(self.btn_new)
        self.toggle_layout.add_widget(self.btn_existing)
        self.add_widget(self.toggle_layout)

        self.content_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=80)
        self.add_widget(self.content_layout)

        self.frames_input = text_input("Frames:", default="frames/**/*.png")
        self.output_dataset_input = text_input("Dataset:", default=default_new_path)
        
        os.makedirs("datasets", exist_ok=True)
        datasets = [f for f in os.listdir("datasets") if f.endswith(".pt")]
        default_existing = datasets[0] if datasets else ""
        self.existing_path_val = default_existing
        self.existing_dropdown = labelled_dropdown(
            "Path:", datasets, default_existing, self._on_existing_change
        )

        self.on_mode_change()

    def _on_existing_change(self, val):
        self.existing_path_val = val

    def on_mode_change(self, *args):
        self.content_layout.clear_widgets()
        if self.is_new:
            self.content_layout.add_widget(self.frames_input.parent)
            self.content_layout.add_widget(self.output_dataset_input.parent)
        else:
            self.content_layout.add_widget(self.existing_dropdown.parent)
            self.content_layout.add_widget(Label(size_hint_y=None, height=36)) # filler

    @property
    def is_new(self):
        return self.btn_new.state == 'down'

    def get_path(self):
        if self.is_new:
            return self.output_dataset_input.text
        else:
            return os.path.join("datasets", self.existing_path_val)

    def get_patterns(self):
        return self.frames_input.text.split()


class CreateDatasetPopup(Popup):
    def __init__(self, on_log=None, **kwargs):
        super().__init__(title="Create Dataset", size_hint=(0.8, 0.9), **kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        self.train_ds_menu = DatasetMenu("Train Dataset", "datasets/train_0.pt")
        layout.add_widget(self.train_ds_menu)
        
        self.test_ds_menu = DatasetMenu("Test Dataset", "datasets/test_0.pt")
        layout.add_widget(self.test_ds_menu)
        
        self.epochs_input = text_input("Epochs:", default="20")
        layout.add_widget(self.epochs_input.parent)
        
        self.output_model_input = text_input("Model:", default="models/0.pth")
        layout.add_widget(self.output_model_input.parent)
        
        action_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        self.train_btn = Button(text="Train", size_hint_x=None, width=100)
        self.train_btn.bind(on_release=self._on_train)
        action_layout.add_widget(self.train_btn)
        
        self.log_label = Label(text="", halign="left", valign="middle", padding=(10, 0))
        self.log_label.bind(size=self.log_label.setter('text_size'))
        action_layout.add_widget(self.log_label)

        self.on_log = on_log
        
        layout.add_widget(action_layout)
        self.content = layout

    def _on_train(self, instance):
        train_path = self.train_ds_menu.get_path()
        test_path = self.test_ds_menu.get_path()
        out_model = self.output_model_input.text
        
        try:
            epochs = int(self.epochs_input.text)
        except ValueError:
            epochs = 20
        
        self.train_btn.disabled = True
        
        code = (
            "from project.dataset import build_dataset\n"
            "from project.model import run_training\n"
        )
        
        if self.train_ds_menu.is_new:
            patterns = self.train_ds_menu.get_patterns()
            code += f"build_dataset({patterns!r}, output_path={train_path!r})\n"
            
        if self.test_ds_menu.is_new:
            if not self.train_ds_menu.is_new or train_path != test_path:
                patterns = self.test_ds_menu.get_patterns()
                code += f"build_dataset({patterns!r}, output_path={test_path!r})\n"
                
        code += f"run_training(train_dataset={train_path!r}, test_dataset={test_path!r}, model_path={out_model!r}, epochs={epochs})\n"
        
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        def read_output(p):
            for line in iter(p.stdout.readline, ''):
                if line:
                    Clock.schedule_once(lambda dt, l=line.strip(): self._update_log(l))
            p.stdout.close()
            p.wait()
            Clock.schedule_once(lambda dt: self._update_log("Finished!"))
            Clock.schedule_once(lambda dt: setattr(self.train_btn, 'disabled', False))

        threading.Thread(target=read_output, args=(process,), daemon=True).start()

        self.dismiss()

    def _update_log(self, text):
        self.log_label.text = text
        if self.on_log:
            self.on_log(text)


@dataclass(frozen=True, kw_only=True)
class Frame:
    data: MatLike
    time: datetime.datetime
    notes: set[Note] = field(default_factory=set)

class RecordingApp(App):
    frame_buffer: deque[Frame]
    popup: CreateDatasetPopup | None

    def __init__(self, target_port: str, video_device: str, **kwargs):
        super().__init__(**kwargs)
        self.video_device = video_device
        self.midi_listener = MidiListener(target_port)
        self.cap = None
        self.frame_n = 0
        self.frame_buffer = deque(maxlen=30)
        self.scheduled_save = None
        self.pending_notes = []
        self.recording_enabled = False
        self.replace_enabled = True
        self.saved_items = 0
        self.saved_history = []
        self.selected_hand = "right_hand"
        self.selected_pressed_keys = "auto"
        self.selected_fingers = "1"
        self.popup = None

        self.cap = video_capture(self.video_device)
        if not self.cap.isOpened():
            print(f"Camera device '{self.video_device}' is invalid.")
            exit()

        self.update_frame()

    def build(self):
        os.makedirs("frames", exist_ok=True)

        root_layout = BoxLayout(orientation='vertical')
        
        topbar = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        train_button = Button(text="Train", size_hint_x=None, width=100)
        train_button.bind(on_release=self._open_train_popup)
        self.train_log_label = Label(text="", halign="left", valign="middle", padding=(5, 0))
        self.train_log_label.bind(size=self.train_log_label.setter('text_size'))
        topbar.add_widget(train_button)
        topbar.add_widget(self.train_log_label)
        root_layout.add_widget(topbar)

        layout = BoxLayout(orientation='horizontal')

        self.image_view = ImageView(size_hint=(0.7, 1.0))
        layout.add_widget(self.image_view.build())

        sidebar = BoxLayout(orientation='vertical', size_hint=(0.3, 1.0))

        w, h = 0, 0
        frame = self.frame_buffer[0].data
        if frame is not None:
            h, w, _ = frame.shape
        self.cropping_region = CroppingRegion(default_w=w, default_h=h)
        self.image_view.on_touch = self.cropping_region.push_point
        sidebar.add_widget(self.cropping_region.build())


        self.load_skew_button = Button(
            text="Load last skew",
            size_hint_y=None,
            height=38,
        )
        self.load_skew_button.bind(on_release=self._load_last_skew)
        sidebar.add_widget(self.load_skew_button)

        self.flip_x_cb = labelled_checkbox("Flip X:")
        sidebar.add_widget(self.flip_x_cb.parent)

        self.flip_y_cb = labelled_checkbox("Flip Y:")
        sidebar.add_widget(self.flip_y_cb.parent)

        self.gray_cb = labelled_checkbox("Grayscale:")
        sidebar.add_widget(self.gray_cb.parent)

        self.recording_button = Button(
            text="Recording: OFF",
            size_hint_y=None,
            height=42,
        )
        self.recording_button.bind(on_release=self._toggle_recording)
        sidebar.add_widget(self.recording_button)

        self.replace_button = Button(
            text="Replace: ON",
            size_hint_y=None,
            height=38,
        )
        self.replace_button.bind(on_release=self._toggle_replace)
        sidebar.add_widget(self.replace_button)

        self.hand_dropdown = labelled_dropdown(
            "Hand:",
            ["right_hand", "left_hand"],
            self.selected_hand,
            self._update_hand,
            max_height=120,
        )
        sidebar.add_widget(self.hand_dropdown.parent)

        self.pressed_keys_dropdown = labelled_dropdown(
            "Keys:",
            ["auto", "1", "2", "3", "4", "5"],
            self.selected_pressed_keys,
            self._update_pressed_keys,
            max_height=220,
        )
        sidebar.add_widget(self.pressed_keys_dropdown.parent)

        self.fingers_dropdown = labelled_dropdown(
            "Fingers:",
            finger_index_options(),
            self.selected_fingers,
            self._update_fingers,
            max_height=260,
        )
        sidebar.add_widget(self.fingers_dropdown.parent)

        self.saved_counter_label = Label(
            text="Saved images: 0",
            size_hint_y=None,
            height=32,
            halign='left',
            padding=(10, 5),
        )
        self.saved_counter_label.bind(size=self.saved_counter_label.setter('text_size'))
        sidebar.add_widget(self.saved_counter_label)

        self.undo_button = Button(
            text="Undo last save",
            size_hint_y=None,
            height=38,
        )
        self.undo_button.bind(on_release=self._undo_last_save)
        sidebar.add_widget(self.undo_button)

        self.recent_saved_label = Label(
            text="Last 10 dataset items:\nNo items saved in this session.",
            valign='top',
            halign='left',
            padding=(10, 5),
        )
        self.recent_saved_label.bind(size=self.recent_saved_label.setter('text_size'))
        sidebar.add_widget(self.recent_saved_label)

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
        Clock.schedule_interval(self.update, 1.0 / 100.0)

        root_layout.add_widget(layout)

        return root_layout

    def _open_train_popup(self, *_):
        if self.popup is None:
            self.popup = CreateDatasetPopup(on_log=self._update_train_log)
        self.popup.open()

    def update_frame(self) -> Frame | None:
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        now = datetime.datetime.now()
        frame = Frame(data=frame, time=now)
        self.frame_buffer.appendleft(frame)
        self.frame_n += 1

        # Remove frames older than 1 second
        while self.frame_buffer and (now - self.frame_buffer[-1].time).total_seconds() > 1.0:
            self.frame_buffer.pop()

        return frame

    def update(self, dt):
        if not self.cap or not self.cap.isOpened():
            return
        
        t = 1

        frame = self.update_frame()
        if not frame:
            return
        
        self.midi_listener.update()
        frame.notes.update(self.midi_listener.pressed(t))

        time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-5]
        # print(f"{time} Frame {self.frame_n} ({len(self.frame_buffer)} fps), just pressed: {self.midi_listener.just_pressed()}, just released: {self.midi_listener.just_released()}")

        if self.scheduled_save is not None:
            self.scheduled_save += 1
            if self.scheduled_save >= 0:
                self.scheduled_save = None
                self.save_frame(0, prefix="pressed")
        
        if self.midi_listener.just_pressed(t):
            self.save_frame(0)
            self.save_frame(7)
            self.save_frame(-3)

        self._update_camera_view(frame.data)
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

    def _update_train_log(self, t: str):
        self.train_log_label.text = t

    def _update_camera_view(self, frame):
        frame = frame.copy()
        frame = self._transform_frame(frame, outline=True)

        self.image_view.update_image(frame)

    def save_frame(self, index: int, prefix: str | None = None):
        if not self.recording_enabled:
            return
        
        if index < 0:
            self.scheduled_save = index
            return

        frame = self.frame_buffer[index]

        notes = sorted(note.name for note in frame.notes)
        notes_str = "_".join(notes) if notes else "none"
        pressed_keys = (
            len(notes)
            if self.selected_pressed_keys == "auto"
            else int(self.selected_pressed_keys)
        )
        saved_paths = []
        for i in range(0, 3):
            frame_idx = index + 2 - i
            b_frame = self.frame_buffer[frame_idx].data

            path = Path("frames") / self.selected_hand / str(pressed_keys) / self.selected_fingers

            if prefix:
                path /= f"{prefix}_{notes_str}_{i}.png"
            else:
                path /= f"{notes_str}_{i}.png"
            
            path = self._resolve_save_path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            cropped = self._transform_frame(b_frame)
            if cropped is None:
                continue

            if cv2.imwrite(path, cropped):
                saved_paths.append(path)
                print(f"Saved frame {frame_idx} to {path}")
            else:
                print(f"Could not save frame {frame_idx} to {path}")

        if not saved_paths:
            return

        self.saved_items += 1
        self.saved_history.append({
            "label": (
                f"{self.selected_hand}/{pressed_keys}/"
                f"{self.selected_fingers}/{notes_str}"
            ),
            "paths": saved_paths,
        })
        self._save_last_skew_if_changed()

    def _update_sidebar(self):
        self.saved_counter_label.text = f"Saved images: {self.saved_items}"
        self._update_recent_saved_label()

        if not self.pending_notes:
            self.captured_label.text = "Captured: "
            return

        lines = ["Captured: "]
        for _, notes in self.pending_notes:
            lines.append('+'.join(notes))
        self.captured_label.text = ", ".join(lines)

    def _toggle_recording(self, *_):
        self.recording_enabled = not self.recording_enabled
        state = "ON" if self.recording_enabled else "OFF"
        self.recording_button.text = f"Recording: {state}"
        if not self.recording_enabled:
            self.pending_notes.clear()

    def _toggle_replace(self, *_):
        self.replace_enabled = not self.replace_enabled
        state = "ON" if self.replace_enabled else "OFF"
        self.replace_button.text = f"Replace: {state}"

    def _resolve_save_path(self, filepath: Path) -> Path:
        prefix = 0
        while True:
            candidate = filepath.with_name(f"{prefix}__{filepath.stem}{filepath.suffix}")
            if not candidate.exists():
                return candidate
            prefix += 1

    def _undo_last_save(self, *_):
        if not self.saved_history:
            print("No saved items to undo in this session.")
            return

        item = self.saved_history.pop()
        removed = 0
        for path in item["paths"]:
            try:
                if path.exists():
                    path.unlink()
                    removed += 1
            except OSError as exc:
                print(f"Could not remove {path}: {exc}")

        self.saved_items = max(0, self.saved_items - 1)
        self._update_recent_saved_label()
        self.saved_counter_label.text = f"Saved images: {self.saved_items}"
        print(f"Undid {item['label']} ({removed} files removed).")

    def _update_recent_saved_label(self):
        lines = ["Last 10 dataset items:"]
        if not self.saved_history:
            lines.append("No items saved in this session.")
        else:
            recent_items = self.saved_history[-10:]
            lines.extend(item["label"] for item in reversed(recent_items))
        self.recent_saved_label.text = "\n".join(lines)

    def _current_skew_config(self):
        if self.cropping_region.is_rect:
            return None

        points = self.cropping_region.skew_points
        if any(point is None for point in points):
            return None

        return {
            "tl": list(points[0]),
            "tr": list(points[1]),
            "bl": list(points[2]),
            "br": list(points[3]),
        }

    def _read_last_skew_config(self):
        if not LAST_SKEW_CONFIG_PATH.exists():
            return None

        try:
            with LAST_SKEW_CONFIG_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(data, dict):
            return None

        points = []
        for key in ("tl", "tr", "bl", "br"):
            value = data.get(key)
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not all(isinstance(coord, int) for coord in value)
            ):
                return None
            points.append(tuple(value))

        return tuple(points)

    def _save_last_skew_if_changed(self):
        config = self._current_skew_config()
        if config is None:
            return

        current_points = tuple(tuple(config[key]) for key in ("tl", "tr", "bl", "br"))
        if self._read_last_skew_config() == current_points:
            return

        LAST_SKEW_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LAST_SKEW_CONFIG_PATH.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
            file.write("\n")
        print(f"Saved last skew config to {LAST_SKEW_CONFIG_PATH}")

    def _load_last_skew(self, *_):
        points = self._read_last_skew_config()
        if points is None:
            print(f"No valid skew config found at {LAST_SKEW_CONFIG_PATH}")
            return

        self.cropping_region.is_rect = False
        self.cropping_region.skew_points = points
        print(f"Loaded last skew config from {LAST_SKEW_CONFIG_PATH}")

    def _update_hand(self, value: str):
        self.selected_hand = value

    def _update_pressed_keys(self, value: str):
        self.selected_pressed_keys = value

    def _update_fingers(self, value: str):
        self.selected_fingers = value
        
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
