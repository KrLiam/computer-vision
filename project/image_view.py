import cv2
from cv2.typing import MatLike
from kivy.graphics.texture import Texture
from kivy.uix.image import Image


class ImageView:
    def __init__(self, size_hint=(1.0, 1.0), on_touch=None):
        self.image_widget = Image(size_hint=size_hint)
        self._current_image_shape = None
        self.on_touch = on_touch
        self.image_widget.bind(on_touch_down=self._on_touch_down)

    def build(self) -> Image:
        return self.image_widget

    def update_image(self, img: MatLike):
        if img is None or img.size == 0:
            return

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        self._current_image_shape = img.shape[:2]

        buf = cv2.flip(img, 0).tobytes()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture

    def _on_touch_down(self, instance, touch):
        if not instance.collide_point(*touch.pos):
            return False

        if self._current_image_shape is None:
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
            h, w = self._current_image_shape
            if self.on_touch:
                self.on_touch(int(rel_x * w), int(rel_y * h))
            return True
        return False