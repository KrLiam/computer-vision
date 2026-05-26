import glob
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2.typing import MatLike
from time import perf_counter


def show(
    *rows: tuple[MatLike, str] | tuple[tuple[MatLike, str], ...],
    save_path: str | None = None
):
    """
    Displays images in a grid. Each argument is a row.
    A row can be a single (image, title) tuple or a tuple of (image, title) tuples.
    """
    figsize = 5
    processed_rows = []
    for row in rows:
        if isinstance(row[0], (np.ndarray, MatLike)):
            # Single image tuple (MatLike, str)
            processed_rows.append([row])
        else:
            # Tuple of tuples ((MatLike, str), (MatLike, str), ...)
            processed_rows.append(list(row))

    num_rows = len(processed_rows)
    max_cols = max(len(row) for row in processed_rows)

    fig, axes = plt.subplots(
        num_rows,
        max_cols,
        figsize=(max_cols * figsize, num_rows * figsize),
        squeeze=True,
        constrained_layout=True
    )
    plt.tight_layout()

    for r, row in enumerate(processed_rows):
        for c, (img, title) in enumerate(row):
            ax = axes[r, c]
            cmap = 'gray' if len(img.shape) == 2 else None
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
        
        # Hide empty subplots in the row
        for c in range(len(row), max_cols):
            axes[r, c].axis('off')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


def apply_sobel(image: MatLike) -> MatLike:
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    sobel = cv2.magnitude(sobelx, sobely)
    return cv2.convertScaleAbs(sobel)

def apply_canny(image: MatLike) -> MatLike:
    return cv2.Canny(image, 20, 150)

class PerfCounter:
    def __init__(self, label):
        self.label = label
    
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        delta = (perf_counter() - self.time) * 1000
        print(f"{self.label}: {delta:.3f}ms")

def identify_keyboard_adaptive_threshold(
    image: MatLike,
    save_path: str | None = None,
    plot: bool = False,
) -> MatLike:
    pad_size = 61

    with PerfCounter("Identification time"):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 5, 3, 9)

        # Black Key Segmentation: Binarize the image to isolate the black keys
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, 1)

        # Adicionar um padding em todas as bordas
        if pad_size > 0:
            binary = cv2.copyMakeBorder(binary, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)

        # Abertura para remover ruído e manter componentes grandes verticalmente
        kernel = np.ones((21, 1), np.uint8)
        binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Fechamento das brechas entre teclas brancas
        kernel = np.ones((1, 7), np.uint8)
        binary_closed = cv2.morphologyEx(binary_opened, cv2.MORPH_CLOSE, kernel)

        # Analisa contornos com proporção maior que 3:1
        # Considera o contorno com maior área
        contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        countour_areas = []
        for cnt in contours:
            _, _, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            if aspect_ratio < 3:
                continue

            area = cv2.contourArea(cnt)
            countour_areas.append((area, cnt))
        countour_areas.sort(key=lambda el: -el[0])

        selected_component = np.zeros_like(binary_closed)
        if countour_areas:
            _, cnt = countour_areas[0]
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(selected_component, [cnt], -1, 255, thickness=cv2.FILLED)
            # ou opcionalmente preencher o bounding box inteiro:
            # cv2.rectangle(keyboard_mask, (x, y), (x+w, y+h), 255, -1)

        # Fecha possiveis buracos no retângulo principal,
        # incluindo as teclas pretas
        kernel = np.ones((1, 61), np.uint8)
        component_closed = cv2.morphologyEx(selected_component, cv2.MORPH_CLOSE, kernel)

        # Abertura agressiva para manter majoritariamente a área principal
        # do teclado, removendo partes indesejadas que permaneceram até agora
        kernel = np.ones((31, 155), np.uint8)
        component_opened = cv2.morphologyEx(component_closed, cv2.MORPH_OPEN, kernel)

        # Analisa novamente os contornos, pegando o de maior área.
        contours, _ = cv2.findContours(component_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        countour_areas = []
        for cnt in contours:
            _, _, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            if aspect_ratio < 3:
                continue

            area = cv2.contourArea(cnt)
            countour_areas.append((area, cnt))

        countour_areas.sort(key=lambda el: -el[0])
        final_mask = np.zeros_like(binary_closed)
        if countour_areas:
            _, cnt = countour_areas[0]
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            # ou opcionalmente preencher o bounding box inteiro:
            cv2.rectangle(final_mask, (x, y), (x+w, y+h), 255, -1)
    
    # Remove o padding adicionado no início
    unpad = lambda im: im[pad_size:-pad_size, pad_size:-pad_size]
    if pad_size > 0:
        final_mask = unpad(final_mask)
    
    # Recorta a área do teclado da imagem cinza
    cropped_keyboard = cv2.bitwise_and(gray, final_mask)

    if save_path:
        parent_dir = Path(save_path).with_suffix("")
        parent_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(parent_dir / "1_gray.png", gray)
        cv2.imwrite(parent_dir / "2_binary.png", unpad(binary))
        cv2.imwrite(parent_dir / "3_binary_opened.png", unpad(binary_opened))
        cv2.imwrite(parent_dir / "4_binary_closed.png", unpad(binary_closed))
        cv2.imwrite(parent_dir / "5_selected_component.png", unpad(selected_component))
        cv2.imwrite(parent_dir / "6_component_closed.png", unpad(component_closed))
        cv2.imwrite(parent_dir / "7_component_opened.png", unpad(component_opened))
        cv2.imwrite(parent_dir / "8_final_mask.png", final_mask)
        cv2.imwrite(parent_dir / "9_cropped_keyboard.png", unpad(cropped_keyboard))

    if plot:
        show(
            (
                (gray, "Gray"),
                (unpad(binary), "Binary"),
                (unpad(binary_opened), "Binary (Opened)"),
            ),
            (
                (unpad(binary_closed), "Binary (Closed)"),
                (unpad(selected_component), "Selected Component"),
                (unpad(component_closed), "Selected Component (Closed)"),
            ),
            (
                (unpad(component_opened), "Selected Component (Opened)"),
                (final_mask, "Final Keyboard Mask"),
                (cropped_keyboard, "Cropped Keyboard"),
            ),
            save_path=save_path,
        )

    return cropped_keyboard

def debug_keyboard(input: str, save_images: bool = False, plot: bool = False):
    for path in glob.glob(input, recursive=True):
        img = cv2.imread(path, cv2.IMREAD_COLOR_RGB)

        if save_images:
            output = str(Path("ignored_detection") / Path(path))
        else:
            output = None
            plot = True
        
        identify_keyboard_adaptive_threshold(
            img,
            save_path=output,
            plot=plot,
        )