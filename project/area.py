import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2.typing import MatLike

def show(img):
    plt.imshow(img)
    plt.show()

def apply_sobel(image: MatLike) -> MatLike:
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    return cv2.convertScaleAbs(sobel)

def apply_canny(image: MatLike) -> MatLike:
    return cv2.Canny(image, 50, 150)

def preprocess(image: MatLike) -> MatLike:
    # Convert the image to grayscale
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except:
        gray = image
        
    # Apply Canny operator to identify sharp intensity changes
    edges = apply_canny(gray)
    show(edges)
    
    # Use erosion and dilation to clean up the image, remove noise, and connect disjointed segments
    kernel = np.ones((3, 3), np.uint8)
    
    # Morphological closing to connect disjointed segments
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # Morphological opening to remove noise
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    show(cleaned)
    
    return cleaned

def locate(image: MatLike, edges: MatLike) -> MatLike:
    # Apply Hough transform to identify dominant lines (linear orientation)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20)

    show(lines)
    
    median_angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Assume the keyboard is mostly horizontal, filter out vertical lines
            if -45 < angle < 45:
                angles.append(angle)
        if angles:
            median_angle = np.median(angles)
            
    # Rotation and rectification: rotate the frames to align them horizontally
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # To determine ROI, recalculate edges on the rotated image
    try:
        rotated_gray = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2GRAY)
    except:
        rotated_gray = rotated_image
    rotated_edges = apply_canny(rotated_gray)
    
    # Region of Interest (ROI) Selection: project edges horizontally to find the keyboard
    proj = np.sum(rotated_edges, axis=1)
    
    if proj.max() > 0:
        threshold_val = np.max(proj) * 0.3
        y_indices = np.where(proj > threshold_val)[0]
        if len(y_indices) > 0:
            y_min = max(0, y_indices[0] - 20)
            y_max = min(h, y_indices[-1] + 20)
            roi = rotated_image[y_min:y_max, :]
            return roi
            
    return rotated_image

def identify_keys(image: MatLike) -> MatLike:
    # Black Key Segmentation: Binarize the image to isolate the black keys
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except:
        gray = image
        
    # Inverse binary threshold since black keys are dark and white keys are bright
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Rectangular Approximation: Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    black_keys = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # Filter by proportions (specific height-to-width ratios)
        if w > 0:
            ratio = h / w
            if 2.0 < ratio < 10.0 and area > 50:
                black_keys.append((x, y, w, h))
                
    # Sort black keys left-to-right
    black_keys = sorted(black_keys, key=lambda b: b[0])
    
    # Pattern Recognition: identify specific octaves or keys using sets of 2 and 3
    groups = []
    if len(black_keys) > 0:
        current_group = [black_keys[0]]
        for i in range(1, len(black_keys)):
            prev_key = black_keys[i-1]
            curr_key = black_keys[i]
            
            # Gap distance between current and previous black key
            dist = curr_key[0] - (prev_key[0] + prev_key[2])
            avg_w = (prev_key[2] + curr_key[2]) / 2.0
            
            # Gap larger than ~1.5x black key width means transition between 2-group and 3-group
            if dist > avg_w * 1.5:
                groups.append(current_group)
                current_group = [curr_key]
            else:
                current_group.append(curr_key)
        groups.append(current_group)
        
    output_image = image.copy()
    for group in groups:
        if len(group) == 2:
            color = (255, 0, 0)
            label = "C/D/E"
        elif len(group) == 3:
            color = (0, 255, 0)
            label = "F/G/A/B"
        else:
            color = (0, 0, 255)
            label = "?"
            
        for k in group:
            x, y, w, h = k
            cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(output_image, label, (x, max(y - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
    return output_image

def detect_keyboard(image: MatLike) -> MatLike:
    edges = preprocess(image)
    roi = locate(image, edges)
    result = identify_keys(roi)
    return result

if __name__ == "__main__":
    img = cv2.imread("frames_original/1/C2_0.png")
    show(img)
    result = detect_keyboard(img)
    show(result)