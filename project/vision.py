import os
import datetime
import torch
from collections import deque
import cv2
import numpy as np
from cv2.typing import MatLike

from project.crop import CroppingRegion
from project.dataset import frames_to_tensor
from project.midi import format_note, get_note_code, guess_key_positions, tensor_to_notes
from project.model import DEVICE, NeuralNetwork, load_model
from project.record import labelled_checkbox, labelled_dropdown

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

from project.image_view import ImageView

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

        self.build(initial_frame)
        if initial_frame is not None:
            self._update_camera_view(initial_frame)
            
        self._load_current_model()
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

        self.flip_x_cb = labelled_checkbox("Flip X:")
        sidebar.add_widget(self.flip_x_cb.parent)

        self.flip_y_cb = labelled_checkbox("Flip Y:")
        sidebar.add_widget(self.flip_y_cb.parent)

        self.log_label = Label(
            text="Pressed:\nAction Log:\n",
            valign='top',
            halign='left',
            padding=(10, 10)
        )
        self.log_label.bind(size=self.log_label.setter('text_size'))
        sidebar.add_widget(self.log_label)
        self.add_widget(sidebar)
        
    def update(self, dt, frame_data: MatLike):
        self.frame_buffer.append(frame_data)
        self._update_camera_view(frame_data)
        self._test_model()
        self._update_sidebar()

    def _update_camera_view(self, frame):
        frame = frame.copy()

        if self.flip_y_cb.active:
            frame = cv2.flip(frame, 0)

        if self.flip_x_cb.active:
            frame = cv2.flip(frame, 1)

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
                f"Pressed: {', '.join(sorted(pressed))}\n" +
                "Action Log:\n" + '\n'.join(self.log_lines)
            )
    
    def _test_model(self):
        if len(self.frame_buffer) < 3:
            return
        
        frames = []
        for frame in self.frame_buffer:
            frame = self.cropping_region.apply(frame)
            if frame is not None:
                frames.append(frame)
        if not frames:
            return

        x = frames_to_tensor(frames)

        c, h, w, *_ = tuple(x.shape)
        if c != 3 or h != 128 or w != 640:
            return
        
        # convert to (1, c, h, w)
        x = x.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(x.to(DEVICE))
            pred = torch.sigmoid(pred).squeeze(0)

            notes = tensor_to_notes(pred)
            self.detected_notes = [format_note(note) for note in notes]

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
