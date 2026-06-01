from dataclasses import dataclass
import glob
import os

import cv2
import numpy as np

# Disable Kivy's argument parser to avoid conflicts with our own argparse
os.environ["KIVY_NO_ARGS"] = "1"
from kivy.config import Config
Config.set('graphics', 'width', '1280')
Config.set('graphics', 'height', '720')

from cv2.typing import MatLike
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton

from project.image_view import ImageView


@dataclass
class ImageEntry:
    path: str

    @property
    def image(self) -> MatLike:
        return cv2.imread(self.path)


def text_input(label: str, changed = None, default: str = ""):
    box = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
    box.add_widget(Label(text=label, size_hint_x=0.3))
    input = TextInput(text=default, multiline=False, size_hint_x=0.7)
    if changed is not None:
        input.bind(text=changed)
    box.add_widget(input)
    return input

def labelled_checkbox(label: str, active: bool = False):
    box = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
    box.add_widget(Label(text=label, size_hint_x=0.3))
    checkbox = CheckBox(active=active, size_hint_x=0.7)
    box.add_widget(checkbox)
    return checkbox


class CroppingRegion:
    def __init__(self, on_change=None, default_w=0, default_h=0, desired_size=(640, 128)):
        self.on_change = on_change
        self.default_w = default_w
        self.default_h = default_h
        self.desired_size = desired_size
        self._point_idx = 0

    def build(self) -> BoxLayout:
        layout = BoxLayout(orientation='vertical', size_hint_y=None)
        layout.bind(minimum_height=layout.setter('height'))

        layout.add_widget(Label(text="Cropping Region", size_hint_y=None, height=30, bold=True))

        method_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
        method_box.add_widget(Label(text="Method:", size_hint_x=0.3))
        self.btn_rect = ToggleButton(text="Rect", group="method", state="down")
        self.btn_skew = ToggleButton(text="Skew", group="method")
        self.btn_rect.bind(on_press=self.on_method_change)
        self.btn_skew.bind(on_press=self.on_method_change)
        method_box.add_widget(self.btn_rect)
        method_box.add_widget(self.btn_skew)
        layout.add_widget(method_box)

        self.points_container = BoxLayout(orientation='vertical', size_hint_y=None, height=60)
        layout.add_widget(self.points_container)

        w, h = self.default_w, self.default_h
        self.p1_input = text_input(label="P1 (x, y):", default=f"0, 0", changed=self._on_text_change)
        self.p2_input = text_input(label="P2 (x, y):", default=f"{w}, {h}", changed=self._on_text_change)
        self.tl_input = text_input(label="TL (x, y):", default=f"0, 0", changed=self._on_text_change)
        self.tr_input = text_input(label="TR (x, y):", default=f"{w}, 0", changed=self._on_text_change)
        self.bl_input = text_input(label="BL (x, y):", default=f"0, {w}", changed=self._on_text_change)
        self.br_input = text_input(label="BR (x, y):", default=f"{w}, {h}", changed=self._on_text_change)

        self.output_size_label = Label(text="Output size: 0, 0", size_hint_y=None, height=30)
        layout.add_widget(self.output_size_label)

        self.btn_adjust = Button(text="Adjust Size", size_hint_y=None, height=30)
        self.btn_adjust.bind(on_press=self.on_adjust_size)
        layout.add_widget(self.btn_adjust)

        self.on_method_change()
        return layout

    def _on_text_change(self, *args):
        w, h = self.output_size
        self.output_size_label.text = f"Output size: {w}, {h}"

        if self.on_change:
            self.on_change()

    def _parse_point(self, text):
        try:
            parts = text.split(",")
            if len(parts) == 2:
                return int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            pass
        return None

    def _format_point(self, point):
        if point is None:
            return ""
        return f"{point[0]}, {point[1]}"

    def on_method_change(self, *args):
        self._point_idx = 0
        self.points_container.clear_widgets()
        if self.is_rect:
            self.points_container.add_widget(self.p1_input.parent)
            self.points_container.add_widget(self.p2_input.parent)
            self.points_container.height = 60
        else:
            self.points_container.add_widget(self.tl_input.parent)
            self.points_container.add_widget(self.tr_input.parent)
            self.points_container.add_widget(self.bl_input.parent)
            self.points_container.add_widget(self.br_input.parent)
            self.points_container.height = 120
        self._on_text_change()

    @property
    def is_rect(self) -> bool:
        return self.btn_rect.state == 'down'

    @is_rect.setter
    def is_rect(self, value: bool):
        if value:
            self.btn_rect.state = 'down'
            self.btn_skew.state = 'normal'
        else:
            self.btn_rect.state = 'normal'
            self.btn_skew.state = 'down'
        self.on_method_change()

    @property
    def rect_points(self) -> tuple:
        p1 = self._parse_point(self.p1_input.text)
        p2 = self._parse_point(self.p2_input.text)
        return p1, p2

    @rect_points.setter
    def rect_points(self, points: tuple):
        p1, p2 = points
        self.p1_input.text = self._format_point(p1)
        self.p2_input.text = self._format_point(p2)

    @property
    def skew_points(self) -> tuple:
        tl = self._parse_point(self.tl_input.text)
        tr = self._parse_point(self.tr_input.text)
        bl = self._parse_point(self.bl_input.text)
        br = self._parse_point(self.br_input.text)
        return tl, tr, bl, br

    @skew_points.setter
    def skew_points(self, points: tuple):
        tl, tr, bl, br = points
        self.tl_input.text = self._format_point(tl)
        self.tr_input.text = self._format_point(tr)
        self.bl_input.text = self._format_point(bl)
        self.br_input.text = self._format_point(br)
    
    def push_point(self, x: int, y: int):
        if self.is_rect:
            inputs = [self.p1_input, self.p2_input]
        else:
            inputs = [self.tl_input, self.tr_input, self.br_input, self.bl_input]
            
        inputs[self._point_idx].text = self._format_point((x, y))
        self._point_idx = (self._point_idx + 1) % len(inputs)
    
    def set_corners(self, points: list[tuple[int, int]]):
        br, bl, tl, tr = points

        if self.is_rect:
            self.rect_points = (tl, br)
        else:
            self.skew_points = (tl, tr, bl, br)

    def on_adjust_size(self, *args):
        target_w, target_h = self.desired_size
        out_w, out_h = self.output_size

        if out_w == 0 or out_h == 0:
            return

        if self.is_rect:
            p1, p2 = self.rect_points
            if not p1 or not p2:
                return
            
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            new_x1 = cx - target_w / 2.0
            new_x2 = cx + target_w / 2.0
            new_y1 = cy - target_h / 2.0
            new_y2 = cy + target_h / 2.0
            
            min_x, max_x = new_x1, new_x2
            min_y, max_y = new_y1, new_y2
            
            dx = 0
            if self.default_w > 0 and max_x > self.default_w: 
                dx = self.default_w - max_x
            if min_x + dx < 0: 
                dx = -min_x
                
            dy = 0
            if self.default_h > 0 and max_y > self.default_h: 
                dy = self.default_h - max_y
            if min_y + dy < 0: 
                dy = -min_y
            
            self.rect_points = (int(new_x1 + dx), int(new_y1 + dy)), (int(new_x2 + dx), int(new_y2 + dy))
            
        else:
            pts = self.skew_points
            if any(p is None for p in pts):
                return
                
            pts = np.array(pts, dtype=np.float32)
            
            # Calculate local scale factors using the quadrilateral's axes to handle
            # any orientation (including vertical). Applied iteratively to guarantee
            # exact convergence for non-parallelogram shapes.
            for _ in range(10):
                width_top = np.linalg.norm(pts[0] - pts[1])
                width_bot = np.linalg.norm(pts[3] - pts[2])
                current_w = max(width_top, width_bot)
                
                height_left = np.linalg.norm(pts[0] - pts[2])
                height_right = np.linalg.norm(pts[1] - pts[3])
                current_h = max(height_left, height_right)
                
                if current_w < 1e-3 or current_h < 1e-3:
                    break
                    
                if abs(current_w - target_w) == 0 and abs(current_h - target_h) == 0:
                    break
                
                sx = target_w / float(current_w)
                sy = target_h / float(current_h)
                
                tl, tr, bl, br = pts
                C = (tl + tr + bl + br) / 4.0
                
                # W points roughly left to right, H points top to bottom
                W = (tr + br - tl - bl) / 4.0
                H = (bl + br - tl - tr) / 4.0
                
                # Transformation matrix to scale along local W and H axes
                M = np.column_stack((W, H))
                if abs(np.linalg.det(M)) < 1e-6:
                    break
                    
                inv_M = np.linalg.inv(M)
                S = np.array([[sx, 0], [0, sy]])
                A = M @ S @ inv_M
                
                for i in range(4):
                    pts[i] = C + A @ (pts[i] - C)
                    
            min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
            min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
            
            dx = 0
            if self.default_w > 0 and max_x > self.default_w: 
                dx = self.default_w - max_x
            if min_x + dx < 0: 
                dx = -min_x
                
            dy = 0
            if self.default_h > 0 and max_y > self.default_h: 
                dy = self.default_h - max_y
            if min_y + dy < 0: 
                dy = -min_y
            
            self.skew_points = tuple((int(round(p[0] + dx)), int(round(p[1] + dy))) for p in pts)

    def draw_outline(self, img: MatLike):
        if self.is_rect:
            p1, p2 = self.rect_points
            
            if p1 and p2:
                cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        else:
            tl, tr, bl, br = self.skew_points
            
            if tl and tr and bl and br:
                pts = np.array([tl, tr, br, bl], np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)

    @property
    def output_size(self) -> tuple[int, int]:
        if self.is_rect:
            p1, p2 = self.rect_points
            if p1 and p2:
                return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
            return 0, 0
        else:
            tl, tr, bl, br = self.skew_points
            if tl and tr and bl and br:
                src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
                
                width_top = np.linalg.norm(src_pts[0] - src_pts[1])
                width_bot = np.linalg.norm(src_pts[3] - src_pts[2])
                out_w = max(int(width_top), int(width_bot))
                
                height_left = np.linalg.norm(src_pts[0] - src_pts[3])
                height_right = np.linalg.norm(src_pts[1] - src_pts[2])
                out_h = max(int(height_left), int(height_right))
                
                return out_w, out_h
            return 0, 0

    def apply(self, img: MatLike) -> MatLike:
        if self.is_rect:
            p1, p2 = self.rect_points

            if not p1 or not p2:
                print("Invalid points. Cannot apply crop.")
                return

            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])

            if x1 == x2 or y1 == y2:
                print("Invalid cropping region area.")
                return

            h, w, _ = img.shape
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            return img[cy1:cy2, cx1:cx2]
        else:
            tl, tr, bl, br = self.skew_points

            if not tl or not tr or not bl or not br:
                print("Invalid points. Cannot apply crop.")
                return

            crop_pts = [tl, tr, br, bl]
            src_pts = np.array(crop_pts, dtype=np.float32)
            
            out_w, out_h = self.output_size
            
            dst_pts = np.array([
                [0, 0],
                [out_w - 1, 0],
                [out_w - 1, out_h - 1],
                [0, out_h - 1]
            ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            return cv2.warpPerspective(img, M, (out_w, out_h))

    def inverse_apply(self, img: MatLike, target_shape: tuple[int, int]) -> MatLike | None:
        target_h, target_w = target_shape[:2]
        
        if self.is_rect:
            p1, p2 = self.rect_points

            if not p1 or not p2:
                return None

            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])

            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(target_w, x2), min(target_h, y2)
            mh, mw = cy2 - cy1, cx2 - cx1

            if mh <= 0 or mw <= 0:
                return None
            
            img_resized = cv2.resize(img, (mw, mh))
            res = np.zeros((target_h, target_w, *img.shape[2:]), dtype=img.dtype)
            res[cy1:cy2, cx1:cx2] = img_resized
            return res
        else:
            tl, tr, bl, br = self.skew_points

            if not tl or not tr or not bl or not br:
                return None

            crop_pts = [tl, tr, br, bl]
            dst_pts = np.array(crop_pts, dtype=np.float32)
            
            out_w, out_h = self.output_size
            
            src_pts = np.array([
                [0, 0],
                [out_w - 1, 0],
                [out_w - 1, out_h - 1],
                [0, out_h - 1]
            ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            return cv2.warpPerspective(img, M, (target_w, target_h))


class CroppingApp(App):
    images: list[ImageEntry]

    def __init__(self, patterns: list[str], **kwargs):
        super().__init__(**kwargs)
        self.patterns = patterns
        self.images = []
        self.is_cropping = False
        self.current_image_index = 0
        
        for pattern in self.patterns:
            for path in glob.glob(pattern, recursive=True):
                self.images.append(ImageEntry(path))

        if not self.images:
            print("No images found matching the given patterns.")
            self.reference_image = None
        else:
            self.reference_image = self.images[0].image

    def build(self):
        Window.bind(on_keyboard=self.on_keyboard)
        layout = BoxLayout(orientation='horizontal')

        # Left: Frame Image
        self.image_view = ImageView(size_hint=(0.7, 1.0))
        layout.add_widget(self.image_view.build())

        # Right: Sidebar
        sidebar = BoxLayout(orientation='vertical', size_hint=(0.3, 1.0), padding=10)

        # Determine default P2 based on the dimensions of the reference image
        h, w = (0, 0)
        if self.reference_image is not None:
            h, w, _ = self.reference_image.shape

        self.cropping_region = CroppingRegion(on_change=self.update_preview, default_w=w, default_h=h)
        self.image_view.on_touch = self.cropping_region.push_point
        sidebar.add_widget(self.cropping_region.build())

        # Flip X
        self.flip_x_cb = labelled_checkbox("Flip X:")
        sidebar.add_widget(self.flip_x_cb.parent)

        # Flip Y
        self.flip_y_cb = labelled_checkbox("Flip Y:")
        sidebar.add_widget(self.flip_y_cb.parent)

        # Grayscale Checkbox
        self.gray_cb = labelled_checkbox("Grayscale:")
        sidebar.add_widget(self.gray_cb.parent)

        # Apply Button
        apply_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        self.apply_btn = Button(text="Apply", size_hint_x=None, width=100)
        self.apply_btn.bind(on_press=self.apply_crop)
        apply_box.add_widget(self.apply_btn)
        apply_box.add_widget(Label())  # Filler to push button left
        sidebar.add_widget(apply_box)

        # Images section
        sidebar.add_widget(Label(text="Images", size_hint_y=None, height=30, bold=True))

        self.image_list_layout = BoxLayout(orientation='vertical', size_hint_y=None, spacing=2)
        self.image_list_layout.bind(minimum_height=self.image_list_layout.setter('height'))

        self.image_buttons = []
        for i, entry in enumerate(self.images):
            btn = ToggleButton(text=entry.path, group='images', size_hint_y=None, height=30)
            btn.bind(on_press=lambda instance, idx=i: self.select_image(idx))
            self.image_list_layout.add_widget(btn)
            self.image_buttons.append(btn)

        self.scroll_view = ScrollView(size_hint=(1.0, 1.0))
        self.scroll_view.add_widget(self.image_list_layout)
        sidebar.add_widget(self.scroll_view)

        layout.add_widget(sidebar)

        if self.reference_image is not None:
            self.select_image(0)

        return layout

    def on_keyboard(self, window, key, scancode, codepoint, modifier):
        if self.is_cropping:
            return False
            
        if key == 273:  # Up arrow
            self.select_image(self.current_image_index - 1)
            return True
        elif key == 274:  # Down arrow
            self.select_image(self.current_image_index + 1)
            return True
            
        return False

    def select_image(self, idx):
        if not (0 <= idx < len(self.images)):
            return
        self.current_image_index = idx
        for i, btn in enumerate(self.image_buttons):
            if i == idx:
                btn.state = 'down'
                if hasattr(self, 'scroll_view'):
                    self.scroll_view.scroll_to(btn)
            else:
                btn.state = 'normal'
                
        self.reference_image = self.images[idx].image
        self.update_preview()

    def update_preview(self, *args):
        if getattr(self, 'is_cropping', False):
            return
        if self.reference_image is None:
            return

        img_copy = self.reference_image.copy()
        self.cropping_region.draw_outline(img_copy)

        self.show_image(img_copy)

    def show_image(self, img):
        self.image_view.update_image(img)

    def apply_crop(self, *args):
        if self.is_cropping:
            return
            
        self.is_cropping = True
        self.apply_btn.disabled = True
        
        self.crop_gen = self.cropping_task()
        Clock.schedule_interval(self.pump_crop_gen, 0.0)

    def pump_crop_gen(self, dt):
        try:
            next(self.crop_gen)
            return True
        except StopIteration:
            return False
            
    def cropping_task(self):
        total = len(self.images)

        for idx, entry in enumerate(self.images):
            self.select_image(idx)
            img = self.reference_image
            if img is None:
                continue
                
            cropped = self.cropping_region.apply(img)

            if self.flip_y_cb.active:
                frame = cv2.flip(frame, 0)

            if self.flip_x_cb.active:
                frame = cv2.flip(frame, 1)
                
            if self.gray_cb.active:
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(entry.path, cropped)

            self.apply_btn.text = f"{idx + 1}/{total}"
            
            self.show_image(cropped)
            yield
            
        print("Done cropping.")
        self.stop()


def run_cropping(patterns: list[str]):
    CroppingApp(patterns).run()
