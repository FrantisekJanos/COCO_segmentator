import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QFrame, QColorDialog, QScrollArea, QLineEdit, QRadioButton, QButtonGroup
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen
from PyQt5.QtCore import Qt
import numpy as np
from skimage import io, feature, color, measure, morphology

class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Segmentační anotátor (PyQt5)')
        self.setGeometry(100, 100, 1000, 700)
        self.image = None
        self.edges = None
        self.contours = []  # seznam všech kontur
        self.selected_segments = {}  # index: label
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Image display
        self.image_label = QLabel('Nahrajte obrázek')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.mousePressEvent = self.image_clicked

        # Load button
        load_btn = QPushButton('Nahrát obrázek')
        load_btn.clicked.connect(self.load_image)

        # Sliders for edge detection
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setMinimum(1)
        self.sigma_slider.setMaximum(30)
        self.sigma_slider.setValue(2)
        self.sigma_slider.setTickInterval(1)
        self.sigma_slider.valueChanged.connect(self.update_edges)

        self.low_slider = QSlider(Qt.Horizontal)
        self.low_slider.setMinimum(0)
        self.low_slider.setMaximum(100)
        self.low_slider.setValue(10)
        self.low_slider.setTickInterval(1)
        self.low_slider.valueChanged.connect(self.update_edges)

        self.high_slider = QSlider(Qt.Horizontal)
        self.high_slider.setMinimum(0)
        self.high_slider.setMaximum(100)
        self.high_slider.setValue(30)
        self.high_slider.setTickInterval(1)
        self.high_slider.valueChanged.connect(self.update_edges)

        # Slider labels
        self.sigma_label = QLabel('Sigma: 2')
        self.low_label = QLabel('Low threshold: 10')
        self.high_label = QLabel('High threshold: 30')

        self.sigma_slider.valueChanged.connect(lambda v: self.sigma_label.setText(f'Sigma: {v}'))
        self.low_slider.valueChanged.connect(lambda v: self.low_label.setText(f'Low threshold: {v}'))
        self.high_slider.valueChanged.connect(lambda v: self.high_label.setText(f'High threshold: {v}'))

        # Výběr barvy
        self.color = QColor(255, 0, 0)
        self.color_btn = QPushButton('Barva čáry')
        self.color_btn.setStyleSheet(f'background-color: {self.color.name()};')
        self.color_btn.clicked.connect(self.choose_color)

        # Slider pro tloušťku čáry
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setMinimum(1)
        self.thickness_slider.setMaximum(10)
        self.thickness_slider.setValue(2)
        self.thickness_slider.setTickInterval(1)
        self.thickness_slider.valueChanged.connect(self.update_edges)
        self.thickness_label = QLabel('Tloušťka čáry: 2')
        self.thickness_slider.valueChanged.connect(lambda v: self.thickness_label.setText(f'Tloušťka čáry: {v}'))

        # Morfologický slider
        self.morph_slider = QSlider(Qt.Horizontal)
        self.morph_slider.setMinimum(0)
        self.morph_slider.setMaximum(20)
        self.morph_slider.setValue(0)
        self.morph_slider.setTickInterval(1)
        self.morph_slider.valueChanged.connect(self.update_edges)
        self.morph_label = QLabel('Spojení kontur: 0')
        self.morph_slider.valueChanged.connect(lambda v: self.morph_label.setText(f'Spojení kontur: {v}'))
        # Výběrový režim
        self.mode_group = QButtonGroup()
        self.contour_radio = QRadioButton('Výběr kontury')
        self.area_radio = QRadioButton('Výběr oblasti')
        self.contour_radio.setChecked(True)
        self.mode_group.addButton(self.contour_radio)
        self.mode_group.addButton(self.area_radio)
        self.contour_radio.toggled.connect(self.update_edges)
        self.area_radio.toggled.connect(self.update_edges)

        # Panel pro miniatury vybraných segmentů
        self.selection_panel = QWidget()
        self.selection_layout = QVBoxLayout()
        self.selection_panel.setLayout(self.selection_layout)
        self.selection_scroll = QScrollArea()
        self.selection_scroll.setWidgetResizable(True)
        self.selection_scroll.setWidget(self.selection_panel)
        self.selection_scroll.setMinimumWidth(180)
        self.selection_scroll.setMaximumWidth(250)
        self.selection_scroll.setWindowTitle('Vybrané objekty')

        # Layouts
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.sigma_label)
        slider_layout.addWidget(self.sigma_slider)
        slider_layout.addWidget(self.low_label)
        slider_layout.addWidget(self.low_slider)
        slider_layout.addWidget(self.high_label)
        slider_layout.addWidget(self.high_slider)
        slider_layout.addWidget(self.thickness_label)
        slider_layout.addWidget(self.thickness_slider)
        slider_layout.addWidget(self.color_btn)
        slider_layout.addWidget(self.morph_label)
        slider_layout.addWidget(self.morph_slider)
        slider_layout.addWidget(self.contour_radio)
        slider_layout.addWidget(self.area_radio)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(load_btn)
        controls_layout.addLayout(slider_layout)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(controls_layout)
        left_layout.addWidget(self.image_label)
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.selection_scroll)

        main_widget.setLayout(main_layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Vyberte obrázek', '', 'Obrázky (*.png *.jpg *.jpeg *.bmp)')
        if file_name:
            self.image = io.imread(file_name)
            if self.image.ndim == 2:
                rgb = color.gray2rgb(self.image)
            else:
                rgb = self.image
            self.display_image(rgb)
            self.update_edges()

    def display_image(self, img_np):
        # Convert numpy image to QImage and display
        if img_np.dtype != np.uint8:
            img_np = (255 * (img_np / img_np.max())).astype(np.uint8)
        h, w, ch = img_np.shape
        bytes_per_line = ch * w
        qimg = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_edges(self):
        if self.image is None:
            return
        sigma = self.sigma_slider.value()
        low = self.low_slider.value() / 100.0
        high = self.high_slider.value() / 100.0
        morph_size = self.morph_slider.value()
        if low >= high:
            self.display_image(self.image)
            self.image_label.setText('Low threshold musí být menší než High threshold!')
            return
        gray = color.rgb2gray(self.image)
        edges = feature.canny(gray, sigma=sigma, low_threshold=low, high_threshold=high)
        # Morfologické spojení kontur
        if morph_size > 0:
            selem = morphology.disk(morph_size)
            edges = morphology.binary_closing(edges, selem)
        self.edges = edges
        from skimage import measure
        self.contours = measure.find_contours(edges, 0.5)
        # Připrav QPixmap pro vykreslení
        if self.image.dtype != np.uint8:
            img_np = (255 * (self.image / self.image.max())).astype(np.uint8)
        else:
            img_np = self.image.copy()
        if img_np.ndim == 2:
            img_np = color.gray2rgb(img_np)
        h, w, ch = img_np.shape
        qimg = QImage(img_np.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        painter = QPainter(pixmap)
        # Nejprve vykresli všechny kontury základní barvou
        pen = QPen(self.color, self.thickness_slider.value())
        painter.setPen(pen)
        for idx, contour in enumerate(self.contours):
            if idx in self.selected_segments:
                continue  # vybrané vykreslíme zvlášť
            if len(contour) < 2:
                continue
            for i in range(len(contour) - 1):
                y1, x1 = contour[i]
                y2, x2 = contour[i + 1]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        # Zvýrazni vybrané segmenty zeleně
        highlight_pen = QPen(QColor(0, 255, 0), self.thickness_slider.value() + 1)
        painter.setPen(highlight_pen)
        for idx in self.selected_segments:
            contour = self.contours[idx]
            if len(contour) < 2:
                continue
            for i in range(len(contour) - 1):
                y1, x1 = contour[i]
                y2, x2 = contour[i + 1]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        painter.end()
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.update_selection_panel()

    def update_selection_panel(self):
        # Smaž staré widgety
        while self.selection_layout.count():
            child = self.selection_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        if not self.selected_segments:
            label = QLabel('Žádné objekty nejsou vybrány.')
            self.selection_layout.addWidget(label)
            return
        for idx in self.selected_segments:
            contour = self.contours[idx]
            # Vytvoř miniaturu segmentu
            thumb = self.create_segment_thumbnail(contour)
            thumb_label = QLabel()
            thumb_label.setPixmap(thumb)
            thumb_label.setFixedSize(80, 80)
            # Pole pro label
            label_edit = QLineEdit()
            label_edit.setPlaceholderText('Label')
            label_edit.setText(self.selected_segments[idx])
            label_edit.textChanged.connect(lambda text, i=idx: self.set_segment_label(i, text))
            # Tlačítko pro odebrání
            remove_btn = QPushButton('Odebrat')
            remove_btn.clicked.connect(lambda _, i=idx: self.remove_segment(i))
            # Layout pro jeden segment
            seg_layout = QVBoxLayout()
            seg_layout.addWidget(thumb_label)
            seg_layout.addWidget(label_edit)
            seg_layout.addWidget(remove_btn)
            seg_widget = QWidget()
            seg_widget.setLayout(seg_layout)
            self.selection_layout.addWidget(seg_widget)
        self.selection_layout.addStretch(1)

    def create_segment_thumbnail(self, contour):
        # Najdi bounding box
        y_min, x_min = np.floor(contour.min(axis=0)).astype(int)
        y_max, x_max = np.ceil(contour.max(axis=0)).astype(int)
        pad = 5
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(self.image.shape[0], y_max + pad)
        x_max = min(self.image.shape[1], x_max + pad)
        crop = self.image[y_min:y_max, x_min:x_max]
        # Vytvoř masku segmentu
        mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        from skimage.draw import polygon_perimeter
        rr, cc = polygon_perimeter(contour[:, 0] - y_min, contour[:, 1] - x_min, mask.shape, clip=True)
        mask[rr, cc] = 1
        # Zvýrazni segment v ořezu
        if crop.ndim == 2:
            crop_rgb = color.gray2rgb(crop)
        else:
            crop_rgb = crop.copy()
        crop_rgb[mask == 1] = [0, 255, 0]
        # Převod na QPixmap
        if crop_rgb.dtype != np.uint8:
            crop_rgb = (255 * (crop_rgb / crop_rgb.max())).astype(np.uint8)
        h, w, ch = crop_rgb.shape
        qimg = QImage(crop_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        return pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def set_segment_label(self, idx, text):
        self.selected_segments[idx] = text

    def remove_segment(self, idx):
        if idx in self.selected_segments:
            del self.selected_segments[idx]
            self.update_edges()

    def choose_color(self):
        color = QColorDialog.getColor(initial=self.color, parent=self)
        if color.isValid():
            self.color = color
            self.color_btn.setStyleSheet(f'background-color: {self.color.name()};')
            self.update_edges()

    def image_clicked(self, event):
        if self.image is None or not self.contours:
            return
        try:
            # Přepočet kliknutí na souřadnice originálu
            label_w = self.image_label.width()
            label_h = self.image_label.height()
            pixmap = self.image_label.pixmap()
            if pixmap is None:
                return
            pm_w = pixmap.width()
            pm_h = pixmap.height()
            x = event.pos().x() - (label_w - pm_w) // 2
            y = event.pos().y() - (label_h - pm_h) // 2
            if not (0 <= x < pm_w and 0 <= y < pm_h):
                return
            # Přepočet na originální rozlišení
            img_h, img_w = self.image.shape[:2]
            scale = min(label_w / img_w, label_h / img_h)
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            if self.area_radio.isChecked():
                # Výběr oblasti (connected component)
                from skimage import measure
                labeled = measure.label(self.edges, connectivity=2)
                if not (0 <= orig_y < labeled.shape[0] and 0 <= orig_x < labeled.shape[1]):
                    print('Kliknutí mimo obrázek')
                    return
                label_val = labeled[orig_y, orig_x]
                if label_val == 0:
                    print('Kliknutí mimo segment')
                    return
                # Najdi všechny kontury této oblasti
                mask = (labeled == label_val)
                region_contours = measure.find_contours(mask, 0.5)
                if not region_contours:
                    print('V oblasti nebyla nalezena kontura')
                    return
                contour = max(region_contours, key=lambda c: c.shape[0])
                # Najdi index této kontury v self.contours (nebo přidej novou)
                idx = None
                for i, c in enumerate(self.contours):
                    if np.allclose(c, contour, atol=2):
                        idx = i
                        break
                if idx is None:
                    self.contours.append(contour)
                    idx = len(self.contours) - 1
                if idx in self.selected_segments:
                    del self.selected_segments[idx]
                else:
                    self.selected_segments[idx] = ''
                self.update_edges()
            else:
                # Výběr kontury (nejbližší segment)
                min_dist = float('inf')
                min_idx = None
                for idx, contour in enumerate(self.contours):
                    dists = np.sqrt((contour[:, 1] - orig_x) ** 2 + (contour[:, 0] - orig_y) ** 2)
                    dist = dists.min()
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = idx
                if min_idx is not None and min_dist < 10:  # tolerance 10 px
                    if min_idx in self.selected_segments:
                        del self.selected_segments[min_idx]
                    else:
                        self.selected_segments[min_idx] = ''
                    self.update_edges()
        except Exception as e:
            print(f'Chyba při výběru oblasti: {e}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageAnnotator()
    window.show()
    sys.exit(app.exec_()) 