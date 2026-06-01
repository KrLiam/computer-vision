import os
import os
import datetime
import json
import torch
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from cv2.typing import MatLike

from project.area import find_corners, identify_keyboard_adaptive_threshold
from project.crop import CroppingRegion
from project.dataset import frames_to_tensor
from project.midi import format_note, get_note_code, guess_key_positions
from project.model import DEVICE, NeuralNetwork, load_model
from project.record import labelled_checkbox, labelled_dropdown, text_input

# Disable Kivy's argument parser to avoid conflicts with our own argparse
os.environ["KIVY_NO_ARGS"] = "1"
from kivy.config import Config
Config.set('graphics', 'width', '1280')
Config.set('graphics', 'height', '720')
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.slider import Slider

from project.image_view import ImageView

LAST_SKEW_CONFIG_PATH = Path("frames") / "last_skew.json"


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

class VisionContainer(BoxLayout):
    model: NeuralNetwork
    frame_buffer: deque[MatLike]
    detected_notes: list[str]
    previous_notes: set[str]
    log_lines: deque[str]


    def __init__(self, model_path: str | None, initial_frame: MatLike | None, **kwargs):
        super().__init__(orientation='horizontal', padding=(5, 10), **kwargs)
        self.model = NeuralNetwork().to(DEVICE)
        self.model.eval()

        self.model_path = model_path
        self.last_mod_time = None
        os.makedirs("models", exist_ok=True)
        self.frame_buffer = deque(maxlen=3)
        self.detected_notes = []
        self.previous_notes = set()
        self.log_lines = deque(maxlen=10)
        self.status_text = "Waiting for frames"
        self.is_frozen = False
        self.frozen_frame = None

        self.build(initial_frame)
        if initial_frame is not None:
            self._update_camera_view(initial_frame)
            
        self._load_current_model()
        self._load_last_skew()
        Clock.schedule_interval(self._check_model_modified, 1.0)

    def _load_current_model(self):
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path, self.model)
                self.model.eval()
                mtime = os.path.getmtime(self.model_path)
                self.last_mod_time = mtime
                dt_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y/%m/%d %H:%M:%S")
                if hasattr(self, 'model_mod_time_label'):
                    self.model_mod_time_label.text = f"Model last modified at {dt_str}"
            except Exception as e:
                print(f"Failed to load model {self.model_path}: {e}")
                self.last_mod_time = None
                if hasattr(self, 'model_mod_time_label'):
                    self.model_mod_time_label.text = "Error loading model"
        else:
            self.last_mod_time = None
            if hasattr(self, 'model_mod_time_label'):
                self.model_mod_time_label.text = "Model not found or none selected"

    def _check_model_modified(self, dt):
        if self.model_path and os.path.exists(self.model_path):
            current_mtime = os.path.getmtime(self.model_path)
            if self.last_mod_time is None or current_mtime > self.last_mod_time:
                self._load_current_model()
        

    def build(self, initial_frame: MatLike | None):
        self.image_view = ImageView(size_hint=(0.7, 1.0))
        self.add_widget(self.image_view.build())

        sidebar = BoxLayout(orientation='vertical', size_hint=(0.3, 1.0))

        models = [f for f in os.listdir("models") if f.endswith(".pth")]
        initial_model_name = os.path.basename(self.model_path) if self.model_path else ""
        if initial_model_name not in models and models:
            initial_model_name = models[0]
            self.model_path = os.path.join("models", initial_model_name)
            
        model_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=36)
        self.model_dropdown = labelled_dropdown(
            "Model:",
            models,
            initial_model_name,
            self._on_model_select
        )
        model_layout.add_widget(self.model_dropdown.parent)
        
        refresh_btn = Button(text="Refresh", size_hint_x=0.3)
        refresh_btn.bind(on_release=self._on_refresh_models)
        model_layout.add_widget(refresh_btn)
        
        sidebar.add_widget(model_layout)
        
        self.model_mod_time_label = Label(
            text="Model last modified at ---",
            size_hint_y=None, height=30,
            halign="left", valign="middle"
        )
        self.model_mod_time_label.bind(size=self.model_mod_time_label.setter('text_size'))
        sidebar.add_widget(self.model_mod_time_label)

        w, h = 0, 0
        if initial_frame is not None:
            h, w, _ = initial_frame.shape

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

        self.freeze_button = Button(
            text="Freeze image",
            size_hint_y=None,
            height=38,
        )
        self.freeze_button.bind(on_release=self._toggle_freeze)
        sidebar.add_widget(self.freeze_button)

        self.flip_x_cb = labelled_checkbox("Flip X:")
        sidebar.add_widget(self.flip_x_cb.parent)

        self.flip_y_cb = labelled_checkbox("Flip Y:")
        sidebar.add_widget(self.flip_y_cb.parent)

        self.grayscale_cb = labelled_checkbox("Grayscale:")
        sidebar.add_widget(self.grayscale_cb.parent)

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

        self.threshold_input = text_input("Threshold:", default="0.5")
        sidebar.add_widget(self.threshold_input.parent)

        self.log_label = Label(
            text="Pressed:\nAction Log:\n",
            valign='top',
            halign='left',
            padding=(10, 10)
        )
        self.log_label.bind(size=self.log_label.setter('text_size'))
        sidebar.add_widget(self.log_label)
        self.add_widget(sidebar)

        Clock.schedule_interval(self.update_keyboard_area, 0.5)

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

    def _load_last_skew(self, *_):
        points = self._read_last_skew_config()
        if points is None:
            self.status_text = "No saved skew config"
            return

        self.cropping_region.is_rect = False
        self.cropping_region.skew_points = points
        self.status_text = "Loaded last skew"
        
    def update(self, dt, frame_data: MatLike):
        raw_frame = frame_data
        if self.is_frozen:
            if self.frozen_frame is None:
                self.frozen_frame = frame_data.copy()
            raw_frame = self.frozen_frame

        frame = self._transform_camera_frame(raw_frame)

        self.frame_buffer.append(frame)
        self._update_camera_view(frame)
        self._test_model()
        self._update_sidebar()

    def update_keyboard_area(self):
        frame = self.frame_buffer[-1]
        mask = identify_keyboard_adaptive_threshold(frame)
        points = find_corners(mask)
        self.cropping_region.set_corners(points)

    def _transform_camera_frame(self, frame):
        frame = frame.copy()

        exposure_scale = 2 ** self.exposure_slider.value
        contrast = self.contrast_slider.value
        brightness = self.brightness_slider.value
        frame = np.clip(frame.astype(np.float32) * exposure_scale * contrast + brightness, 0, 255).astype(np.uint8)

        if self.grayscale_cb.active:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if self.flip_y_cb.active:
            frame = cv2.flip(frame, 0)

        if self.flip_x_cb.active:
            frame = cv2.flip(frame, 1)

        return frame

    def _update_camera_view(self, frame):
        frame = frame.copy()

        out_w, out_h = self.cropping_region.output_size
        if out_w > 0 and out_h > 0 and self.detected_notes:
            mask = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            key_positions = guess_key_positions((out_w, out_h))
            
            for note_str in self.detected_notes:
                code = get_note_code(note_str)
                if code in key_positions:
                    x, y, w, h = key_positions[code]
                    cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 255), -1)
                    
            warped_mask = self.cropping_region.inverse_apply(mask, frame.shape)
            if warped_mask is not None:
                alpha = 0.75
                active = warped_mask > 0
                frame[active] = (frame[active] * (1 - alpha) + warped_mask[active] * alpha).astype(np.uint8)

        self.cropping_region.draw_outline(frame)
        
        self.image_view.update_image(frame)

    def _update_sidebar(self):
        current_notes = set(self.detected_notes)
        pressed = current_notes - self.previous_notes
        released = self.previous_notes - current_notes

        if pressed or released:
            time_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-5]  # Truncates to 1 decimal place (e.g., 14:32:01.4)

            if pressed:
                notes_str = " + ".join(sorted(pressed))
                self.log_lines.append(f"[{time_str}] Pressed {notes_str}")
            if released:
                notes_str = " + ".join(sorted(released))
                self.log_lines.append(f"[{time_str}] Released {notes_str}")

            self.previous_notes = current_notes

        self.log_label.text = (
            f"Pressed: {', '.join(sorted(current_notes))}\n"
            f"Status: {self.status_text}\n"
            "Action Log:\n" + '\n'.join(self.log_lines)
        )

    def _threshold(self):
        try:
            return float(self.threshold_input.text.replace(",", "."))
        except ValueError:
            return 0.5

    def _reset_camera_params(self, *_):
        self.brightness_slider.value = 0
        self.exposure_slider.value = 0
        self.contrast_slider.value = 1
        self.grayscale_cb.active = False

    def _toggle_freeze(self, *_):
        self.is_frozen = not self.is_frozen
        if self.is_frozen:
            self.frozen_frame = None
            self.freeze_button.text = "Unfreeze image"
            self.status_text = "Image frozen"
        else:
            self.frozen_frame = None
            self.freeze_button.text = "Freeze image"
            self.status_text = "Image live"
    
    def _test_model(self):
        if len(self.frame_buffer) < 3:
            self.status_text = f"Waiting for frames ({len(self.frame_buffer)}/3)"
            return
        
        frames = []
        for frame in self.frame_buffer:
            frame = self.cropping_region.apply(frame)
            if frame is not None:
                frames.append(frame)
        if len(frames) < 3:
            self.status_text = "Select or load a valid crop region"
            return

        try:
            x = frames_to_tensor(frames)
        except Exception as error:
            self.status_text = f"Model input error: {error}"
            return

        c, h, w, *_ = tuple(x.shape)
        if c != 3 or h != 128 or w != 640:
            self.status_text = f"Invalid input shape: {tuple(x.shape)}"
            return
        
        # convert to (1, c, h, w)
        x = x.unsqueeze(0)

        with torch.no_grad():
            try:
                pred = self.model(x.to(DEVICE))
                pred = torch.sigmoid(pred).squeeze(0)
            except Exception as error:
                self.status_text = f"Model test error: {error}"
                return

            threshold = self._threshold()
            top_values, top_indices = torch.topk(pred, k=min(5, pred.numel()))
            top_notes = [
                f"{format_note(idx.item() + 36)}={value.item():.2f}"
                for value, idx in zip(top_values, top_indices)
            ]
            notes = [
                idx + 36
                for idx, value in enumerate(pred.detach().cpu())
                if value.item() > threshold
            ]
            self.detected_notes = [format_note(note) for note in notes]
            self.status_text = f"max {top_values[0].item():.2f} | top {' '.join(top_notes)}"

    def _on_model_select(self, val):
        if val:
            self.model_path = os.path.join("models", val)
        else:
            self.model_path = None
        self._load_current_model()

    def _on_refresh_models(self, *_):
        os.makedirs("models", exist_ok=True)
        models = [f for f in os.listdir("models") if f.endswith(".pth")]
        dropdown = self.model_dropdown.dropdown
        dropdown.clear_widgets()
        for option in models:
            item = Button(text=option, size_hint_y=None, height=36)
            item.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(item)
        
        if self.model_dropdown.text not in models:
            if models:
                self.model_dropdown.text = models[0]
                self._on_model_select(models[0])
            else:
                self.model_dropdown.text = ""
                self._on_model_select("")
        else:
            self._load_current_model()


    def on_stop(self):
        pass


def run_vision(model_path: str, camera: str):
    from project.record import UnifiedApp
    UnifiedApp(target_port=None, video_device=camera, model_path=model_path, initial_mode="Testing").run()
