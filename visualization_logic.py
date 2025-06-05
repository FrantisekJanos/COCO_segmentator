import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PyQt5.QtCore import QRect, QPoint
from skimage import measure


def np_to_qimage(np_img):
    """
    Convert a numpy RGB or grayscale image to QImage.
    """
    if np_img.ndim == 2:
        h, w = np_img.shape
        bytes_per_line = w
        return QImage(np_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
    elif np_img.shape[2] == 3:
        h, w, ch = np_img.shape
        bytes_per_line = ch * w
        return QImage(np_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    elif np_img.shape[2] == 4:
        h, w, ch = np_img.shape
        bytes_per_line = ch * w
        return QImage(np_img.data, w, h, bytes_per_line, QImage.Format_RGBA8888).copy()
    else:
        raise ValueError("Unsupported image shape for QImage conversion.")


def draw_segmentation_overlay(
    base_image: np.ndarray,
    segmentations: list,
    alpha: float = 0.4,
    color_map=None,
    draw_labels=True,
    label_font=None
) -> QPixmap:
    """
    Vykreslí overlay segmentací na obrázek.
    segmentations: list of dicts, každý má 'mask' (2D bool/uint8), 'label', 'color' (QColor or tuple), optional 'polygon', optional 'outline_width'
    """
    qimage = np_to_qimage(base_image)
    pixmap = QPixmap.fromImage(qimage)
    painter = QPainter(pixmap)
    try:
        if label_font:
            painter.setFont(label_font)

        for idx, seg in enumerate(segmentations):
            mask = seg.get('mask')  # 2D numpy array
            color = seg.get('color')
            outline_width = seg.get('outline_width', 4)
            if color is None and color_map:
                color = color_map(idx)
            if color is None:
                color = QColor.fromHsv((idx * 37) % 360, 255, 255, int(255 * alpha))
            elif isinstance(color, tuple):
                color = QColor(*color)
            else:
                color = QColor(color)
            
            # Draw mask overlay
            if mask is not None:
                overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                overlay[mask > 0] = [color.red(), color.green(), color.blue(), int(255 * alpha)]
                overlay_img = np_to_qimage(overlay)
                painter.drawImage(0, 0, overlay_img)
            # Draw polygon/outline if present
            polygon = seg.get('polygon')
            if polygon is not None and len(polygon) > 1:
                pen = QPen(color, outline_width)
                painter.setPen(pen)
                points = [QPoint(int(x), int(y)) for x, y in polygon]
                for i in range(len(points) - 1):
                    painter.drawLine(points[i], points[i + 1])
                # Optionally close polygon
                if np.allclose(polygon[0], polygon[-1]):
                    painter.drawLine(points[-1], points[0])
            # Pokud není polygon, vykresli outline přes kontury masky
            elif mask is not None:
                contours = measure.find_contours(mask, 0.5)
                pen = QPen(color, outline_width)
                painter.setPen(pen)
                for contour in contours:
                    points = [QPoint(int(x), int(y)) for y, x in contour]
                    for i in range(len(points) - 1):
                        painter.drawLine(points[i], points[i + 1])
            # Draw label
            if draw_labels and 'label' in seg:
                label = seg['label']
                # Najít centroid masky nebo polygonu
                if mask is not None:
                    ys, xs = np.where(mask)
                    if len(xs) > 0:
                        cx, cy = int(xs.mean()), int(ys.mean())
                    else:
                        cx, cy = 10, 10
                elif polygon is not None:
                    xs, ys = zip(*polygon)
                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                else:
                    cx, cy = 10, 10
                painter.setPen(QPen(QColor(0, 0, 0, 200), 2))
                painter.drawText(cx + 1, cy + 1, label)
                painter.setPen(QPen(QColor(255, 255, 255, 255), 1))
                painter.drawText(cx, cy, label)
    finally:
        painter.end()
    return pixmap 