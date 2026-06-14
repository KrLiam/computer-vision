from dataclasses import dataclass, field
import datetime
import json
import os
import random
import subprocess
import sys
import re
import logging
import queue
from pathlib import Path
import threading
from collections import deque

import cv2
import numpy as np
from cv2.typing import MatLike
import mido

from project.area import find_corners, identify_keyboard_adaptive_threshold
from project.crop import CroppingRegion, labelled_checkbox, text_input
from project.dataset import EXPECTED_IMAGE_SIZE, build_dataset
from project.model import run_training

# Disable Kivy's argument parser to avoid conflicts with our own argparse
os.environ["KIVY_NO_ARGS"] = "1"
from kivy.config import Config
Config.set('graphics', 'width', '1280')
Config.set('graphics', 'height', '720')

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton

from project.midi import MidiListener, Note
from project.image_view import ImageView

logging.getLogger('PIL').setLevel(logging.WARNING)

LAST_SKEW_CONFIG_PATH = Path("frames") / "last_skew.json"
PRESETS_DIR = Path("presets")
NONE_SAVE_PROBABILITY = 0.15
SAVE_QUEUE_MAX_SIZE = 20


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


def labelled_slider(
    label: str,
    minimum: float,
    maximum: float,
    default: float,
    step: float,
    value_format: str = "{:.0f}",
):
    box = BoxLayout(orientation='horizontal', size_hint_y=None, height=36)
    box.add_widget(Label(text=label, size_hint_x=0.32))

    slider = Slider(min=minimum, max=maximum, value=default, step=step, size_hint_x=0.5)
    value_label = Label(text=value_format.format(default), size_hint_x=0.18)
    slider.bind(value=lambda _, value: setattr(value_label, "text", value_format.format(value)))

    box.add_widget(slider)
    box.add_widget(value_label)
    return slider


def finger_index_options() -> list[str]:
    options = []
    for mask in range(1, 1 << 5):
        fingers = [str(idx + 1) for idx in range(5) if mask & (1 << idx)]
        options.append("-".join(fingers))
    return options


def format_seconds_for_filename(seconds: float) -> str:
    if seconds <= 0:
        seconds = 1.0
    if seconds.is_integer():
        return str(int(seconds))
    return f"{seconds:.2f}".rstrip("0").rstrip(".")


class DatasetMenu(BoxLayout):
    def __init__(self, title: str, default_new_path: str, **kwargs):
        super().__init__(orientation='vertical', size_hint_y=None, height=210, **kwargs)
        self.on_change = None
        self.size_error_count = 0
        self.image_count = 0
        self.add_widget(Label(text=title, size_hint_y=None, height=30, bold=True))

        self.toggle_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
        self.btn_create = ToggleButton(text="Create", group=f"ds_{title}", state="down")
        self.btn_existing = ToggleButton(text="Existing", group=f"ds_{title}")
        self.btn_create.bind(on_press=self.on_mode_change)
        self.btn_existing.bind(on_press=self.on_mode_change)
        self.toggle_layout.add_widget(self.btn_create)
        self.toggle_layout.add_widget(self.btn_existing)
        self.add_widget(self.toggle_layout)

        self.content_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=90)
        self.add_widget(self.content_layout)

        self.frames_input = text_input("Frames:", changed=self._on_frames_change, default="frames/**/*.png")
        self.output_dataset_input = text_input("Dataset:", default=default_new_path)
        self.cap_none_cb = labelled_checkbox("Cap none:", active=True)
        self.cap_none_cb.bind(active=self._on_cap_none_change)
        
        self.info_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
        self.samples_label = Label(text="Dataset will be built on Train", halign="left", valign="middle")
        self.samples_label.bind(size=self.samples_label.setter('text_size'))
        self.info_layout.add_widget(self.samples_label)
        
        self.update_existing_datasets()

        self.on_mode_change()

    def update_existing_datasets(self):
        os.makedirs("datasets", exist_ok=True)
        self.datasets = [f for f in os.listdir("datasets") if f.endswith(".pt") or f.endswith(".tar")]

        default_existing = self.datasets[0] if self.datasets else ""
        self.existing_path_val = default_existing

        self.existing_dropdown = labelled_dropdown(
            "Path:", self.datasets, default_existing, self._on_existing_change
        )

    def _on_existing_change(self, val):
        self.existing_path_val = val
        self._notify_change()

    def on_mode_change(self, *args):
        self.content_layout.clear_widgets()

        self.update_existing_datasets()

        if self.is_create:
            self.content_layout.add_widget(self.frames_input.parent)
            self.content_layout.add_widget(self.cap_none_cb.parent)
            self.content_layout.add_widget(self.output_dataset_input.parent)
            self.content_layout.add_widget(self.info_layout)
            self.content_layout.height = 150
            self.height = 210
            self._update_samples()
        else:
            self.content_layout.add_widget(self.existing_dropdown.parent)
            self.content_layout.add_widget(Label(size_hint_y=None, height=36)) # filler
            self.content_layout.height = 72
            self.height = 132
            self._notify_change()

    @property
    def is_create(self):
        return self.btn_create.state == 'down'

    def _on_frames_change(self, instance, value):
        Clock.unschedule(self._update_samples)
        Clock.schedule_once(self._update_samples, 0.5)

    def _on_cap_none_change(self, instance, value):
        self._update_samples()

    def _update_samples(self, dt=0):
        self.samples = None
        self.image_count = 1 if self.frames_input.text.split() else 0
        self.size_error_count = 0
        self.samples_label.text = "Dataset will be built on Train"
        self._notify_change()
        
    def get_path(self):
        if self.is_create:
            return self.output_dataset_input.text
        else:
            return os.path.join("datasets", self.existing_path_val)

    def get_patterns(self):
        return self.frames_input.text.split()

    def get_cap_none(self):
        return self.cap_none_cb.active

    def has_valid_image_sizes(self):
        return not self.is_create or bool(self.frames_input.text.split())

    def _notify_change(self):
        if self.on_change:
            self.on_change()


class TrainingPopup(Popup):
    def __init__(self, on_log=None, **kwargs):
        super().__init__(title="Training", size_hint=(0.8, 0.9), **kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        self.train_ds_menu = DatasetMenu("Train Dataset", "datasets/6_train.tar")
        layout.add_widget(self.train_ds_menu)
        
        self.test_ds_menu = DatasetMenu("Test Dataset", "datasets/6_test.tar")
        layout.add_widget(self.test_ds_menu)

        layout.add_widget(Label(text="Dataset Split", size_hint_y=None, height=30, bold=True))

        self.auto_split_cb = labelled_checkbox("Auto split:", active=True)
        layout.add_widget(self.auto_split_cb.parent)

        self.test_ratio_input = text_input("Test %:", default="20")
        layout.add_widget(self.test_ratio_input.parent)
        
        layout.add_widget(Label(text="Model", size_hint_y=None, height=30, bold=True))

        self.epochs_input = text_input("Epochs:", default="20")
        layout.add_widget(self.epochs_input.parent)

        self.batch_size_input = text_input("Batch size:", default="32")
        layout.add_widget(self.batch_size_input.parent)
        
        self.output_model_input = text_input("Model:", default="models/6.pth")
        layout.add_widget(self.output_model_input.parent)

        self.start_from_scratch_cb = labelled_checkbox("Start from scratch:")
        layout.add_widget(self.start_from_scratch_cb.parent)
        
        action_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        self.train_btn = Button(text="Train", size_hint_x=None, width=100)
        self.train_btn.bind(on_release=self._on_train)
        action_layout.add_widget(self.train_btn)
        
        self.log_label = Label(text="", halign="left", valign="middle", padding=(10, 0))
        self.log_label.bind(size=self.log_label.setter('text_size'))
        action_layout.add_widget(self.log_label)

        self.on_log = on_log
        self.train_ds_menu.on_change = self._refresh_train_button
        self.test_ds_menu.on_change = self._refresh_train_button
        self._refresh_train_button()
        
        layout.add_widget(action_layout)
        layout.add_widget(Label())  # filler to push items to the top
        self.content = layout

    def _on_train(self, instance):
        if not self._datasets_are_valid():
            self._update_log("All dataset images must be 640x128 before training.")
            return

        train_path = self.train_ds_menu.get_path()
        test_path = self.test_ds_menu.get_path()
        out_model = self.output_model_input.text
        
        try:
            epochs = int(self.epochs_input.text)
        except ValueError:
            epochs = 20

        try:
            batch_size = int(self.batch_size_input.text)
        except ValueError:
            batch_size = 32

        try:
            test_ratio = float(self.test_ratio_input.text.replace(",", "."))
            if test_ratio > 1:
                test_ratio /= 100
        except ValueError:
            test_ratio = 0.2

        auto_split = (
            self.auto_split_cb.active
            and self.train_ds_menu.is_create
            and self.test_ds_menu.is_create
        )

        if auto_split and train_path == test_path:
            self._update_log("Train and test dataset paths must be different for auto split.")
            return
        
        self.train_btn.disabled = True
        
        code = (
            "from project.dataset import build_dataset, build_train_test_datasets\n"
            "from project.model import run_training\n"
        )

        if auto_split:
            patterns = self.train_ds_menu.get_patterns()
            cap_none = self.train_ds_menu.get_cap_none()
            code += (
                "build_train_test_datasets("
                f"{patterns!r}, "
                f"train_output_path={train_path!r}, "
                f"test_output_path={test_path!r}, "
                f"test_ratio={test_ratio!r}, "
                f"cap_none={cap_none!r}"
                ")\n"
            )
        else:
            if self.train_ds_menu.is_create:
                patterns = self.train_ds_menu.get_patterns()
                cap_none = self.train_ds_menu.get_cap_none()
                code += f"build_dataset({patterns!r}, output_path={train_path!r}, cap_none={cap_none!r})\n"

            if self.test_ds_menu.is_create:
                if not self.train_ds_menu.is_create or train_path != test_path:
                    patterns = self.test_ds_menu.get_patterns()
                    cap_none = self.test_ds_menu.get_cap_none()
                    code += f"build_dataset({patterns!r}, output_path={test_path!r}, cap_none={cap_none!r})\n"
                
        start_from_scratch = self.start_from_scratch_cb.active
        code += (
            "run_training("
            f"train_dataset={train_path!r}, "
            f"test_dataset={test_path!r}, "
            f"model_path={out_model!r}, "
            f"batch_size={batch_size}, "
            f"epochs={epochs}, "
            f"start_from_scratch={start_from_scratch!r}"
            ")\n"
        )
        
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
            return_code = p.wait()
            status = "Finished!" if return_code == 0 else f"Failed with exit code {return_code}"
            Clock.schedule_once(lambda dt, s=status: self._update_log(s))
            Clock.schedule_once(lambda dt: setattr(self.train_btn, 'disabled', False))

        threading.Thread(target=read_output, args=(process,), daemon=True).start()

        self.dismiss()

    def _datasets_are_valid(self):
        return (
            self.train_ds_menu.has_valid_image_sizes()
            and self.test_ds_menu.has_valid_image_sizes()
        )

    def _refresh_train_button(self):
        if hasattr(self, "train_btn"):
            self.train_btn.disabled = not self._datasets_are_valid()

    def _update_log(self, text):
        self.log_label.text = text
        if self.on_log:
            self.on_log(text)


@dataclass(frozen=True, kw_only=True)
class Frame:
    data: MatLike
    time: datetime.datetime
    notes: set[Note] = field(default_factory=set)


@dataclass(frozen=True, kw_only=True)
class SaveJob:
    frames: list[tuple[int, MatLike]]
    notes_str: str
    pressed_keys: int
    hand: str
    fingers: str
    prefix: str | None
    perturb_enabled: bool
    perturb_config: dict[str, float]
    transform_config: dict


@dataclass(frozen=True, kw_only=True)
class SaveResult:
    label: str
    paths: list[Path]


class RecordingContainer(BoxLayout):
    frame_buffer: deque[Frame]
    popup: TrainingPopup | None

    def __init__(self, target_port: str | None, initial_frame: MatLike | None, **kwargs):
        super().__init__(orientation='vertical', padding=(5, 10), **kwargs)
        if target_port:
            self.midi_listener = MidiListener(target_port)
            self.midi_listener.start()
        else:
            self.midi_listener = None
        self.frame_n = 0
        self.frame_buffer = deque(maxlen=300)
        self.preset_recording_frames: list[Frame] = []
        self.preset_recording_started_at: datetime.datetime | None = None
        self.scheduled_save = None
        self.pending_notes = []
        self.recording_enabled = False
        self.replace_enabled = True
        self.saved_items = 0
        self.dropped_save_items = 0
        self.saved_history = []
        self.save_queue: queue.Queue[SaveJob | None] = queue.Queue(maxsize=SAVE_QUEUE_MAX_SIZE)
        self.save_results: queue.Queue[SaveResult | Exception] = queue.Queue()
        self.save_worker = threading.Thread(target=self._save_worker_loop, daemon=True)
        self.save_worker.start()
        self.selected_hand = "right_hand"
        self.selected_pressed_keys = "auto"
        self.selected_fingers = "1"
        self.popup = None
        self.train_epoch = "0"
        self.train_accuracy = "--.-%"

        self.build(initial_frame)
        if initial_frame is not None:
            self._update_camera_view(initial_frame)

    def build(self, initial_frame: MatLike | None):
        os.makedirs("frames", exist_ok=True)
        PRESETS_DIR.mkdir(exist_ok=True)

        layout = BoxLayout(orientation='horizontal')

        preset_panel = BoxLayout(orientation='vertical', size_hint=(0.2, 1.0), padding=(0, 0, 5, 0), spacing=6)
        layout.add_widget(preset_panel)

        self.image_view = ImageView(size_hint=(0.5, 1.0))
        layout.add_widget(self.image_view.build())

        sidebar = BoxLayout(orientation='vertical', size_hint=(0.3, 1.0))

        w, h = 0, 0
        if initial_frame is not None:
            h, w, _ = initial_frame.shape
        self.cropping_region = CroppingRegion(default_w=w, default_h=h)
        self.image_view.on_touch = self.cropping_region.push_point
        sidebar.add_widget(self.cropping_region.build())


        #self.load_skew_button = Button(
        #    text="Load last skew",
        #    size_hint_y=None,
        #    height=38,
        #)
        #self.load_skew_button.bind(on_release=self._load_last_skew)
        #sidebar.add_widget(self.load_skew_button)
    
        self.auto_crop_button = Button(
            text="Auto-crop",
            size_hint_y=None,
            height=38,
        )
        self.auto_crop_button.bind(on_release=lambda _: self.update_keyboard_area())
        sidebar.add_widget(self.auto_crop_button)

        self.flip_x_cb = labelled_checkbox("Flip X:")
        sidebar.add_widget(self.flip_x_cb.parent)

        self.flip_y_cb = labelled_checkbox("Flip Y:")
        sidebar.add_widget(self.flip_y_cb.parent)

        self.gray_cb = labelled_checkbox("Grayscale:")
        sidebar.add_widget(self.gray_cb.parent)

        self.brightness_slider = labelled_slider("Brightness:", -100, 100, 0, 1)
        sidebar.add_widget(self.brightness_slider.parent)

        self.exposure_slider = labelled_slider("Exposure:", -3, 3, 0, 0.1, "{:.1f}")
        sidebar.add_widget(self.exposure_slider.parent)

        self.contrast_slider = labelled_slider("Contrast:", 0, 3, 1, 0.05, "{:.2f}")
        sidebar.add_widget(self.contrast_slider.parent)

        reset_camera_button = Button(
            text="Reset camera params",
            size_hint_y=None,
            height=38,
        )
        reset_camera_button.bind(on_release=self._reset_camera_params)
        sidebar.add_widget(reset_camera_button)

        self.perturb_save_cb = labelled_checkbox("Perturb save:", active=True)
        sidebar.add_widget(self.perturb_save_cb.parent)

        self.brightness_delta_input = text_input("Brightness +/-:", default="30")
        sidebar.add_widget(self.brightness_delta_input.parent)

        self.exposure_delta_input = text_input("Exposure +/-:", default="0.2")
        sidebar.add_widget(self.exposure_delta_input.parent)

        self.contrast_delta_input = text_input("Contrast +/-:", default="0.25")
        sidebar.add_widget(self.contrast_delta_input.parent)

        self.perspective_delta_input = text_input("Perspective px:", default="2")
        sidebar.add_widget(self.perspective_delta_input.parent)

        self.crop_offset_input = text_input("Crop offset px:", default="5")
        sidebar.add_widget(self.crop_offset_input.parent)

        recording_buttons = BoxLayout(orientation='horizontal', size_hint_min_y=43)
        self.recording_button = Button(
            text="Recording: OFF",
            size_hint_y=None,
            height=42,
        )
        self.recording_button.bind(on_release=self._toggle_recording)
        recording_buttons.add_widget(self.recording_button)
        self.record_single_button = Button(
            text="Record Single",
            size_hint_y=None,
            size_hint_x=0.5,
            height=42,
        )
        self.record_single_button.bind(on_release=self._record_single)
        recording_buttons.add_widget(self.record_single_button)
        sidebar.add_widget(recording_buttons)

        preset_buttons = BoxLayout(orientation='vertical', size_hint_y=None, height=88, spacing=4)
        self.start_preset_button = Button(
            text="Start preset recording",
            size_hint_y=None,
            height=42,
        )
        self.start_preset_button.bind(on_release=self._start_preset_recording)
        preset_buttons.add_widget(self.start_preset_button)

        self.end_preset_button = Button(
            text="End recording",
            size_hint_y=None,
            height=42,
            disabled=True,
        )
        self.end_preset_button.bind(on_release=self._end_preset_recording)
        preset_buttons.add_widget(self.end_preset_button)
        preset_panel.add_widget(preset_buttons)

        preset_status_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=82, spacing=4)
        self.discard_preset_button = Button(
            text="Discard preset",
            size_hint_y=None,
            height=38,
            disabled=True,
        )
        self.discard_preset_button.bind(on_release=self._discard_preset_recording)
        preset_status_layout.add_widget(self.discard_preset_button)

        self.preset_status_label = Label(
            text="Preset: idle",
            size_hint_y=None,
            height=36,
            halign="left",
            valign="middle",
            padding=(8, 0),
        )
        self.preset_status_label.bind(size=self.preset_status_label.setter('text_size'))
        preset_status_layout.add_widget(self.preset_status_label)
        preset_panel.add_widget(preset_status_layout)

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
            text="Saved items: 0",
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
            text="Last 10 saved items:\nNo items saved in this session.",
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

        self.add_widget(layout)

        bottombar = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        train_button = Button(text="Train", size_hint_x=None, width=100)
        train_button.bind(on_release=self._open_train_popup)
        self.train_log_label = Label(text="", halign="left", valign="middle", padding=(5, 0))
        self.train_log_label.bind(size=self.train_log_label.setter('text_size'))
        bottombar.add_widget(train_button)
        bottombar.add_widget(self.train_log_label)
        self.add_widget(bottombar)


    def _open_train_popup(self, *_):
        if self.popup is None:
            self.popup = TrainingPopup(on_log=self._update_train_log)
        self.popup.open()

    def update(self, dt, frame_data: MatLike):
        now = datetime.datetime.now()
        frame = Frame(data=frame_data, time=now)
        self.frame_buffer.appendleft(frame)
        self.frame_n += 1

        while self.frame_buffer and (now - self.frame_buffer[-1].time).total_seconds() > 1.0:
            self.frame_buffer.pop()
        
        t = 1
        if self.midi_listener:
            self.midi_listener.update()
            frame.notes.update(self.midi_listener.pressed(t))

        if self.preset_recording_started_at is not None:
            self.preset_recording_frames.append(
                Frame(data=frame.data.copy(), time=frame.time, notes=frame.notes.copy())
            )

        time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-5]
        # print(f"{time} Frame {self.frame_n} ({len(self.frame_buffer)} fps), just pressed: {self.midi_listener.just_pressed()}, just released: {self.midi_listener.just_released()}")

        if self.scheduled_save is not None:
            self.scheduled_save += 1
            if self.scheduled_save >= 0:
                self.scheduled_save = None
                self.save_frame(0, prefix="pressed")
        
        if self.midi_listener and self.midi_listener.just_pressed(t):
            self.save_frame(0)
            self.save_frame(7)
            self.save_frame(-3)

        self._update_camera_view(frame.data)
        self._drain_save_results()
        self._update_sidebar()

    def update_keyboard_area(self):
        #if not self.auto_crop:
        #    return
        if not self.frame_buffer:
            return
        frame = self.frame_buffer[-1]
        frame = self._transform_frame_flip(frame.data)
        
        mask = identify_keyboard_adaptive_threshold(frame)
        points = find_corners(mask)
        if not points:
            return
        self.cropping_region.set_corners(points)

    def _transform_frame_color(self, frame):
        exposure_scale = 2 ** self.exposure_slider.value
        contrast = self.contrast_slider.value
        brightness = self.brightness_slider.value
        frame = np.clip(frame.astype(np.float32) * exposure_scale * contrast + brightness, 0, 255).astype(np.uint8)

        if self.gray_cb.active:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        return frame

    def _transform_frame_flip(self, frame):
        if self.flip_y_cb.active:
            frame = cv2.flip(frame, 0)

        if self.flip_x_cb.active:
            frame = cv2.flip(frame, 1)
            
        return frame
    
    def _transform_frame(
        self,
        frame,
        outline: bool = False,
        padding: int = 10,
        offset: tuple[int, int] = (0, 0),
    ):
        frame = self._transform_frame_color(frame)
        frame = self._transform_frame_flip(frame)
            
        if outline:
            self.cropping_region.draw_outline(frame)
        else:
            frame = self.cropping_region.apply(
                frame,
                target_size=EXPECTED_IMAGE_SIZE,
                padding=padding,
                offset=offset,
            )

        return frame

    def _reset_camera_params(self, *_):
        self.brightness_slider.value = 0
        self.exposure_slider.value = 0
        self.contrast_slider.value = 1
        self.gray_cb.active = False

    def _update_train_log(self, t: str):
        epoch_match = re.search(r"^Epoch\s+(\d+)", t)
        if epoch_match:
            self.train_epoch = epoch_match.group(1)
            
        acc_match = re.search(r"Correct:\s*([\d\.]+%)", t)
        if acc_match:
            self.train_accuracy = acc_match.group(1)

        total_epochs = "?"
        if self.popup:
            try:
                total_epochs = str(int(self.popup.epochs_input.text))
            except ValueError:
                total_epochs = "20"
                
        self.train_log_label.text = f"{self.train_epoch}/{total_epochs} | {self.train_accuracy} | {t}"

    def _update_camera_view(self, frame):
        frame = frame.copy()
        frame = self._transform_frame(frame, outline=True)

        self.image_view.update_image(frame)

    def _float_input(self, input_widget: TextInput, default: float = 0.0) -> float:
        try:
            return float(input_widget.text.replace(",", "."))
        except ValueError:
            return default

    def _perturbation_config(self) -> dict[str, float]:
        return {
            "brightness": max(0.0, self._float_input(self.brightness_delta_input)),
            "exposure": max(0.0, self._float_input(self.exposure_delta_input)),
            "contrast": max(0.0, self._float_input(self.contrast_delta_input)),
            "perspective": max(0.0, self._float_input(self.perspective_delta_input)),
            "offset": max(0.0, self._float_input(self.crop_offset_input)),
        }

    def _random_perturbation(self, config: dict[str, float]) -> dict[str, float | np.ndarray]:
        offset = config["offset"]
        perspective_delta = config["perspective"]
        corner_offsets = np.random.uniform(
            -perspective_delta,
            perspective_delta,
            size=(4, 2),
        ).astype(np.float32)

        return {
            "brightness": np.random.uniform(-config["brightness"], config["brightness"]),
            "exposure": np.random.uniform(-config["exposure"], config["exposure"]),
            "contrast": np.random.uniform(-config["contrast"], config["contrast"]),
            "corner_offsets": corner_offsets,
            "offset": (
                int(np.random.uniform(-offset, offset)),
                int(np.random.uniform(-offset, offset))
            )
        }

    def _apply_perturbation(self, frame: MatLike, params: dict[str, float | np.ndarray]) -> MatLike:
        brightness = float(params["brightness"])
        exposure_scale = 2 ** float(params["exposure"])
        contrast = max(0.05, 1.0 + float(params["contrast"]))

        perturbed = np.clip(
            frame.astype(np.float32) * exposure_scale * contrast + brightness,
            0,
            255,
        ).astype(np.uint8)

        corner_offsets = params["corner_offsets"]
        if isinstance(corner_offsets, np.ndarray) and np.any(corner_offsets):
            h, w = perturbed.shape[:2]
            source = np.array(
                [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]],
                dtype=np.float32,
            )
            destination = source + corner_offsets
            matrix = cv2.getPerspectiveTransform(source, destination)
            perturbed = cv2.warpPerspective(
                perturbed,
                matrix,
                (w, h),
                borderMode=cv2.BORDER_REPLICATE,
            )

        return perturbed

    def _transform_snapshot(self) -> dict:
        return {
            "brightness": self.brightness_slider.value,
            "exposure": self.exposure_slider.value,
            "contrast": self.contrast_slider.value,
            "gray": self.gray_cb.active,
            "flip_x": self.flip_x_cb.active,
            "flip_y": self.flip_y_cb.active,
            "is_rect": self.cropping_region.is_rect,
            "rect_points": self.cropping_region.rect_points,
            "skew_points": self.cropping_region.skew_points,
        }

    def save_frame(
        self,
        index: int,
        prefix: str | None = None,
        none_save_probability: float = NONE_SAVE_PROBABILITY
    ):
        if not self.recording_enabled:
            return
        
        if index < 0:
            self.scheduled_save = index
            return

        max_frame_idx = index + 2
        if max_frame_idx >= len(self.frame_buffer):
            print("Skipping save: not enough buffered frames yet.")
            return

        frame = self.frame_buffer[index]

        notes = sorted(note.name for note in frame.notes)
        if not notes and random.random() > none_save_probability:
            return

        notes_str = "_".join(notes) if notes else "none"
        pressed_keys = (
            len(notes)
            if self.selected_pressed_keys == "auto"
            else int(self.selected_pressed_keys)
        )
        job = SaveJob(
            frames=[
                (index + 2 - i, self.frame_buffer[index + 2 - i].data.copy())
                for i in range(3)
            ],
            notes_str=notes_str,
            pressed_keys=pressed_keys,
            hand=self.selected_hand,
            fingers=self.selected_fingers,
            prefix=prefix,
            perturb_enabled=self.perturb_save_cb.active,
            perturb_config=self._perturbation_config(),
            transform_config=self._transform_snapshot(),
        )

        try:
            self.save_queue.put_nowait(job)
        except queue.Full:
            self.dropped_save_items += 1
            print(f"Skipping {notes_str}: save queue is full ({SAVE_QUEUE_MAX_SIZE}).")

    def _save_worker_loop(self):
        while True:
            job = self.save_queue.get()
            if job is None:
                self.save_queue.task_done()
                break

            try:
                result = self._process_save_job(job)
                if result is not None:
                    self.save_results.put(result)
            except Exception as error:
                self.save_results.put(error)
            finally:
                self.save_queue.task_done()

    def _process_save_job(self, job: SaveJob) -> SaveResult | None:
        params = self._random_perturbation(job.perturb_config)
        cropped_frames = []
        for i, (frame_idx, data) in enumerate(job.frames):
            cropped = self._transform_frame_from_config(
                data,
                job.transform_config,
                offset=params["offset"],
            )
            if cropped is None:
                continue
            cropped_frames.append((i, frame_idx, cropped))

        if len(cropped_frames) != 3:
            return None

        variants: list[tuple[str | None, list[tuple[int, int, MatLike]]]] = [(None, cropped_frames)]
        if job.perturb_enabled:
            variants = []
            for variant_idx in range(3):
                params = self._random_perturbation(job.perturb_config)
                augmented_frames = [
                    (i, frame_idx, self._apply_perturbation(cropped, params))
                    for i, frame_idx, cropped in cropped_frames
                ]
                variants.append((f"aug{variant_idx + 1}", augmented_frames))

        saved_paths = []
        for variant_name, frames in variants:
            for i, frame_idx, cropped in frames:
                path = Path("frames") / job.hand / str(job.pressed_keys) / job.fingers

                filename_parts = []
                if variant_name:
                    filename_parts.append(variant_name)
                if job.prefix:
                    filename_parts.append(job.prefix)
                filename_parts.append(job.notes_str)
                filename = "_".join(filename_parts)
                path /= f"{filename}_{i}.png"

                path = self._resolve_save_path(path)
                path.parent.mkdir(parents=True, exist_ok=True)

                if cv2.imwrite(str(path), cropped):
                    saved_paths.append(path)
                    print(f"Saved frame {frame_idx} to {path}")
                else:
                    print(f"Could not save frame {frame_idx} to {path}")

        if not saved_paths:
            return None

        return SaveResult(
            label=(
                f"{job.hand}/{job.pressed_keys}/"
                f"{job.fingers}/{job.notes_str}"
                f"{' x3 perturbed' if job.perturb_enabled else ''}"
            ),
            paths=saved_paths,
        )

    def _transform_frame_from_config(
        self,
        frame: MatLike,
        config: dict,
        padding: int = 10,
        offset: tuple[int, int] = (0, 0),
    ) -> MatLike | None:
        exposure_scale = 2 ** config["exposure"]
        transformed = np.clip(
            frame.astype(np.float32) * exposure_scale * config["contrast"] + config["brightness"],
            0,
            255,
        ).astype(np.uint8)

        if config["gray"]:
            transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
            transformed = cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)

        if config["flip_y"]:
            transformed = cv2.flip(transformed, 0)

        if config["flip_x"]:
            transformed = cv2.flip(transformed, 1)

        return self._crop_from_config(
            transformed,
            config,
            padding=padding,
            offset=offset,
            target_size=EXPECTED_IMAGE_SIZE,
        )

    def _crop_from_config(
        self,
        img: MatLike,
        config: dict,
        padding: int = 0,
        offset: tuple[int, int] = (0, 0),
        target_size: tuple[int, int] | None = None,
    ) -> MatLike | None:
        if config["is_rect"]:
            p1, p2 = config["rect_points"]
            if not p1 or not p2:
                print("Invalid points. Cannot apply crop.")
                return None

            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])

            x1 += offset[0] - padding
            x2 += offset[0] + padding
            y1 += offset[1] - padding
            y2 += offset[1] + padding

            h, w = img.shape[:2]
            if target_size is not None:
                target_w, target_h = target_size
                target_ratio = target_w / target_h
                current_w = x2 - x1
                current_h = y2 - y1
                if current_h > 0:
                    current_ratio = current_w / current_h
                    factor = current_ratio / target_ratio
                    new_h = current_h * factor
                    diff = new_h - current_h
                    y1 -= diff / 2
                    y2 += diff / 2

            cy1, cy2 = int(y1), int(y2)
            cx1, cx2 = int(x1), int(x2)

            pad_top = max(0, -cy1)
            pad_bottom = max(0, cy2 - h)
            pad_left = max(0, -cx1)
            pad_right = max(0, cx2 - w)

            cy1_clip, cy2_clip = max(0, cy1), min(h, cy2)
            cx1_clip, cx2_clip = max(0, cx1), min(w, cx2)

            if cx1_clip >= cx2_clip or cy1_clip >= cy2_clip:
                print("Invalid cropping region area after expansion.")
                return None

            cropped = img[cy1_clip:cy2_clip, cx1_clip:cx2_clip]
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                cropped = cv2.copyMakeBorder(
                    cropped,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )

            if target_size is not None:
                cropped = cv2.resize(cropped, target_size)
            return cropped

        tl, tr, bl, br = config["skew_points"]
        if not tl or not tr or not bl or not br:
            print("Invalid points. Cannot apply crop.")
            return None

        tl = (tl[0] + offset[0] - padding, tl[1] + offset[1] - padding)
        tr = (tr[0] + offset[0] + padding, tr[1] + offset[1] - padding)
        bl = (bl[0] + offset[0] - padding, bl[1] + offset[1] + padding)
        br = (br[0] + offset[0] + padding, br[1] + offset[1] + padding)

        h_img, w_img = img.shape[:2]

        def clip(pt):
            return (max(0, min(w_img, pt[0])), max(0, min(h_img, pt[1])))

        tl = clip(tl)
        tr = clip(tr)
        bl = clip(bl)
        br = clip(br)

        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        width_top = np.linalg.norm(src_pts[0] - src_pts[1])
        width_bot = np.linalg.norm(src_pts[3] - src_pts[2])
        current_w = max(width_top, width_bot)
        height_left = np.linalg.norm(src_pts[0] - src_pts[3])
        height_right = np.linalg.norm(src_pts[1] - src_pts[2])
        current_h = max(height_left, height_right)

        if current_w <= 0 or current_h <= 0:
            print("Invalid cropping region area.")
            return None

        if target_size is not None:
            target_w, target_h = target_size
            target_ratio = target_w / target_h
            current_ratio = current_w / current_h
            factor = current_ratio / target_ratio
            new_h = current_h * factor
            diff = new_h - current_h

            tl = (tl[0], tl[1] - diff / 2)
            tr = (tr[0], tr[1] - diff / 2)
            bl = (bl[0], bl[1] + diff / 2)
            br = (br[0], br[1] + diff / 2)

            src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
            out_w, out_h = int(current_w), int(new_h)
        else:
            out_w, out_h = int(current_w), int(current_h)

        if out_w <= 0 or out_h <= 0:
            print("Invalid cropping region area after expansion.")
            return None

        dst_pts = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cropped = cv2.warpPerspective(img, matrix, (out_w, out_h))

        if target_size is not None:
            cropped = cv2.resize(cropped, target_size)
        return cropped

    def _drain_save_results(self):
        changed = False
        while True:
            try:
                result = self.save_results.get_nowait()
            except queue.Empty:
                break

            if isinstance(result, Exception):
                print(f"Save worker failed: {result}")
            else:
                self.saved_items += 1
                self.saved_history.append({
                    "label": result.label,
                    "paths": result.paths,
                })
                changed = True
            self.save_results.task_done()

        if changed:
            self._save_last_skew_if_changed()

    def _update_sidebar(self):
        self.saved_counter_label.text = (
            f"Saved items: {self.saved_items} | "
            f"Queue: {self.save_queue.qsize()} | "
            f"Dropped: {self.dropped_save_items}"
        )
        self._update_recent_saved_label()
        self._update_preset_status()

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

    def _record_single(self, *_):
        v = self.recording_enabled
        self.recording_enabled = True
        self.save_frame(0, none_save_probability=1.0)
        self.recording_enabled = v

    def _start_preset_recording(self, *_):
        self.preset_recording_frames.clear()
        self.preset_recording_started_at = datetime.datetime.now()
        self.start_preset_button.disabled = True
        self.end_preset_button.disabled = False
        self.discard_preset_button.disabled = False
        self._update_preset_status()
        print("Started preset recording.")

    def _end_preset_recording(self, *_):
        if self.preset_recording_started_at is None:
            return
        source_frames = self.preset_recording_frames.copy()
        self.preset_recording_started_at = None
        self.preset_recording_frames.clear()
        self.start_preset_button.disabled = False
        self.end_preset_button.disabled = True
        self.discard_preset_button.disabled = True
        self._save_preset_recording(source_frames)
        self._update_preset_status()

    def _discard_preset_recording(self, *_):
        count = len(self.preset_recording_frames)
        self.preset_recording_started_at = None
        self.preset_recording_frames.clear()
        self.start_preset_button.disabled = False
        self.end_preset_button.disabled = True
        self.discard_preset_button.disabled = True
        self._update_preset_status("Preset: discarded")
        print(f"Discarded preset recording ({count} frames).")

    def _update_preset_status(self, text: str | None = None):
        if not hasattr(self, "preset_status_label"):
            return

        if text is not None:
            self.preset_status_label.text = text
            return

        if self.preset_recording_started_at is None:
            self.preset_status_label.text = "Preset: idle"
            return

        elapsed = (datetime.datetime.now() - self.preset_recording_started_at).total_seconds()
        self.preset_status_label.text = (
            f"Preset: rec {format_seconds_for_filename(elapsed)}s "
            f"({len(self.preset_recording_frames)} frames)"
        )

    def _save_preset_recording(self, source_frames: list[Frame]):
        if len(source_frames) < 2:
            print("Not enough frames available to save a preset.")
            return

        duration = max(
            0.25,
            (source_frames[-1].time - source_frames[0].time).total_seconds(),
        )
        video_frames = []
        for frame in source_frames:
            cropped = self._transform_frame(frame.data)
            if cropped is None:
                continue
            target_w, target_h = EXPECTED_IMAGE_SIZE
            if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
                cropped = cv2.resize(cropped, (target_w, target_h))
            if len(cropped.shape) == 2:
                cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
            video_frames.append(cropped)

        if len(video_frames) < 2:
            print("Select or load a valid crop region before saving a preset.")
            return

        recorded_notes = {
            note.name
            for frame in source_frames
            for note in frame.notes
        }
        notes = sorted(recorded_notes)
        notes_str = "_".join(notes) if notes else "none"
        pressed_keys = (
            len(notes)
            if self.selected_pressed_keys == "auto"
            else int(self.selected_pressed_keys)
        )
        duration_str = format_seconds_for_filename(duration)
        path = PRESETS_DIR / self.selected_hand / str(pressed_keys) / self.selected_fingers / f"{notes_str}_{duration_str}.mp4"
        path = self._resolve_preset_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fps = max(1.0, len(video_frames) / duration)
        height, width = video_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            print(f"Could not create preset video at {path}")
            return

        for frame in video_frames:
            writer.write(frame)
        writer.release()

        self.saved_items += 1
        self.saved_history.append({
            "label": (
                f"preset {self.selected_hand}/{pressed_keys}/"
                f"{self.selected_fingers}/{notes_str} ({duration_str}s)"
            ),
            "paths": [path],
        })
        self._save_last_skew_if_changed()
        print(f"Saved preset to {path}")

    def _resolve_save_path(self, filepath: Path) -> Path:
        prefix = 0
        while True:
            candidate = filepath.with_name(f"{prefix}__{filepath.stem}{filepath.suffix}")
            if not candidate.exists():
                return candidate
            prefix += 1

    def _resolve_preset_path(self, filepath: Path) -> Path:
        if not filepath.exists():
            return filepath
        return self._resolve_save_path(filepath)

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
        self._update_sidebar()
        print(f"Undid {item['label']} ({removed} files removed).")

    def _update_recent_saved_label(self):
        lines = ["Last 10 saved items:"]
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
        try:
            self.save_queue.put_nowait(None)
        except queue.Full:
            ...
        if self.midi_listener:
            self.midi_listener.stop()


class UnifiedApp(App):
    def __init__(self, target_port: str | None, video_device: str, model_path: str | None = None, initial_mode: str = "Recording", **kwargs):
        super().__init__(title=initial_mode, **kwargs)
        self.target_port = target_port
        self.video_device = video_device
        self.model_path = model_path
        self.initial_mode = initial_mode
        self.cap = video_capture(self.video_device)
        
        if not self.cap.isOpened():
            print(f"Camera device '{self.video_device}' is invalid.")
            sys.exit(1)
            
        ret, frame = self.cap.read()
        self.initial_frame = frame if ret else None

    def build(self):
        root = BoxLayout(orientation='vertical')
        
        toggle_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        self.btn_record = ToggleButton(text="Recording", group="mode", state="down" if self.initial_mode == "Recording" else "normal")
        self.btn_test = ToggleButton(text="Testing", group="mode", state="down" if self.initial_mode == "Testing" else "normal")
        self.btn_record.bind(on_press=self.on_mode_change)
        self.btn_test.bind(on_press=self.on_mode_change)
        toggle_layout.add_widget(self.btn_record)
        toggle_layout.add_widget(self.btn_test)
        root.add_widget(toggle_layout)
        
        self.content_layout = BoxLayout(orientation='vertical')
        root.add_widget(self.content_layout)
        
        from project.vision import VisionContainer
        self.recording_container = RecordingContainer(self.target_port, self.initial_frame)
        self.vision_container = VisionContainer(self.model_path, self.initial_frame)
        
        self.on_mode_change()
        
        Clock.schedule_interval(self.update, 1.0 / 100.0)
        return root

    def on_mode_change(self, *args):
        if self.btn_record.state == 'normal' and self.btn_test.state == 'normal':
            if args and args[0] == self.btn_record:
                self.btn_record.state = 'down'
            else:
                self.btn_test.state = 'down'

        self.content_layout.clear_widgets()
        if self.btn_record.state == 'down':
            self.content_layout.add_widget(self.recording_container)
        else:
            self.vision_container.refresh_presets()
            self.content_layout.add_widget(self.vision_container)

    def update(self, dt):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return
            
        if self.btn_record.state == 'down':
            self.recording_container.update(dt, frame)
        else:
            self.vision_container.update(dt, frame)

    def on_stop(self):
        if self.cap:
            self.cap.release()
        self.recording_container.on_stop()
        self.vision_container.on_stop()


def run_recording(midi_name: str, video_device: str):
    available_ports = mido.get_input_names()
    target_port = next((p for p in available_ports if midi_name in p), None)

    if not target_port:
        print(f"Could not find a MIDI port containing '{midi_name}'.")
        s = ", ".join(available_ports)
        print(f"Available ports: {s}")
        return

    UnifiedApp(target_port, video_device, initial_mode="Recording").run()
