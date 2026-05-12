import os
import datetime
import torch
from collections import deque
import cv2
from cv2.typing import MatLike

from project.crop import CroppingRegion
from project.dataset import frames_to_tensor
from project.midi import format_note, tensor_to_notes
from project.model import DEVICE, NeuralNetwork, load_model
from project.record import labelled_checkbox, video_capture

# Disable Kivy's argument parser to avoid conflicts with our own argparse
os.environ["KIVY_NO_ARGS"] = "1"
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label

from project.image_view import ImageView

class VisionApp(App):
    model: NeuralNetwork
    cap: cv2.VideoCapture
    frame_buffer: deque[MatLike]
    detected_notes: list[str]
    previous_notes: set[str]
    log_lines: deque[str]


    def __init__(self, model: NeuralNetwork, video_device: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.model.eval()

        self.video_device = video_device
        self.cap = None
        self.frame_buffer = deque(maxlen=3)
        self.detected_notes = []
        self.previous_notes = set()
        self.log_lines = deque(maxlen=10)

        self.cap = video_capture(self.video_device)
        frame = self.get_frame()
        if frame is not None:
            self.frame_buffer.append(frame)
        

    def build(self):
        layout = BoxLayout(orientation='horizontal')

        self.image_view = ImageView(size_hint=(0.7, 1.0))
        layout.add_widget(self.image_view.build())

        sidebar = BoxLayout(orientation='vertical', size_hint=(0.3, 1.0))

        w, h = 0, 0
        frame = self.frame_buffer[0]
        if frame is not None:
            h, w, _ = frame.shape

        self.cropping_region = CroppingRegion(default_w=w, default_h=h)
        self.image_view.on_touch = self.cropping_region.push_point
        sidebar.add_widget(self.cropping_region.build())

        self.flip_x_cb = labelled_checkbox("Flip X:")
        sidebar.add_widget(self.flip_x_cb.parent)

        self.flip_y_cb = labelled_checkbox("Flip Y:")
        sidebar.add_widget(self.flip_y_cb.parent)

        self.log_label = Label(
            text="Action Log:\n",
            valign='top',
            halign='left',
            padding=(10, 10)
        )
        self.log_label.bind(size=self.log_label.setter('text_size'))
        sidebar.add_widget(self.log_label)
        layout.add_widget(sidebar)

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
        
    def update(self, dt):
        frame = self.get_frame()
        if frame is None:
            return   

        self.frame_buffer.append(frame)
        self._update_camera_view(frame)
        self._test_model()
        self._update_sidebar()

    def _update_camera_view(self, frame):
        frame = frame.copy()
        self.cropping_region.draw_outline(frame)

        if self.flip_y_cb.active:
            frame = cv2.flip(frame, 0)

        if self.flip_x_cb.active:
            frame = cv2.flip(frame, 1)
        
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
            self.log_label.text = "Action Log:\n" + "\n".join(self.log_lines)
    
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


    def on_stop(self):
        if self.cap:
            self.cap.release()
        print("\nStopped listening to MIDI input.")


def run_vision(model_path: str, camera: str):
    model = load_model(model_path)

    VisionApp(model, camera).run()
