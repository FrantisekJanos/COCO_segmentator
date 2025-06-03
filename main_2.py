import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QFrame, QColorDialog, QScrollArea, QLineEdit, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt
from skimage import io, color, segmentation, img_as_ubyte, draw
from scipy.ndimage import label as ndi_label

class SuperpixelAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Superpixel Segmentační anotátor (PyQt5)')
        self.setGeometry(100, 100, 1200, 800)
        self.image = None
        self.superpixel_edges = None
        self.manual_borders = []  # seznam ručně dokreslených hranic (každá je seznam bodů)
        self.current_border = []
        self.component_labels = None  # maska komponent
        self.selected_component = None  # index vybrané komponenty
        self.highlight_color = QColor(0, 255, 0, 120)
        self.manual_mode = False
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
        self.image_label.mouseMoveEvent = self.image_mouse_move
        self.image_label.mouseReleaseEvent = self.image_mouse_release

        # Load button
        load_btn = QPushButton('Nahrát obrázek')
        load_btn.clicked.connect(self.load_image)

        # Slider for number of superpixels
        self.slic_slider = QSlider(Qt.Horizontal)
        self.slic_slider.setMinimum(20)
        self.slic_slider.setMaximum(500)
        self.slic_slider.setValue(100)
        self.slic_slider.setTickInterval(10)
        self.slic_slider.valueChanged.connect(self.update_superpixels)
        self.slic_label = QLabel(f'Počet oblastí: 100')

        # Barva zvýraznění
        self.color_btn = QPushButton('Barva zvýraznění')
        self.color_btn.setStyleSheet(f'background-color: {self.highlight_color.name()};')
        self.color_btn.clicked.connect(self.choose_color)

        # Režim ručního kreslení
        self.manual_checkbox = QCheckBox('Ruční dělení oblasti čárou')
        self.manual_checkbox.stateChanged.connect(self.toggle_manual_mode)

        # Panel pro ruční hranice
        self.border_panel = QWidget()
        self.border_layout = QVBoxLayout()
        self.border_panel.setLayout(self.border_layout)
        self.border_scroll = QScrollArea()
        self.border_scroll.setWidgetResizable(True)
        self.border_scroll.setWidget(self.border_panel)
        self.border_scroll.setMinimumWidth(180)
        self.border_scroll.setMaximumWidth(250)
        self.border_scroll.setWindowTitle('Ruční hranice')

        # Layouts
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.slic_label)
        slider_layout.addWidget(self.slic_slider)
        slider_layout.addWidget(load_btn)
        slider_layout.addWidget(self.color_btn)
        slider_layout.addWidget(self.manual_checkbox)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(slider_layout)
        left_layout.addWidget(self.image_label)
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.border_scroll)

        main_widget.setLayout(main_layout)

    def choose_color(self):
        color = QColorDialog.getColor(initial=self.highlight_color, parent=self)
        if color.isValid():
            self.highlight_color = color
            self.color_btn.setStyleSheet(f'background-color: {self.highlight_color.name()};')
            self.display_image()

    def toggle_manual_mode(self, state):
        self.manual_mode = bool(state)
        self.current_border = []
        self.display_image()

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Vyberte obrázek', '', 'Obrázky (*.png *.jpg *.jpeg *.bmp)')
        if file_name:
            self.image = io.imread(file_name)
            if self.image.ndim == 2:
                self.image = color.gray2rgb(self.image)
            self.manual_borders = []
            self.current_border = []
            self.selected_component = None
            self.update_superpixels()

    def update_superpixels(self):
        if self.image is None:
            return
        n_segments = self.slic_slider.value()
        self.slic_label.setText(f'Počet oblastí: {n_segments}')
        segments = segmentation.slic(self.image, n_segments=n_segments, compactness=10, start_label=1)
        # Vytvoř binární masku hranic superpixelů
        edges = segmentation.find_boundaries(segments, mode='thick')
        self.superpixel_edges = edges
        self.selected_component = None
        self.recompute_components()
        self.display_image()
        self.update_border_panel()

    def recompute_components(self):
        if self.image is None:
            return
        # Kombinace hranic superpixelů a ručních čar
        border_mask = self.superpixel_edges.copy()
        for path in self.manual_borders:
            if len(path) > 1:
                for i in range(len(path) - 1):
                    rr, cc = draw.line(path[i][1], path[i][0], path[i+1][1], path[i+1][0])
                    border_mask[rr, cc] = True
        # Invertuj hranice, aby oblasti byly True
        area_mask = ~border_mask
        labels, _ = ndi_label(area_mask)
        self.component_labels = labels

    def display_image(self):
        if self.image is None or self.component_labels is None:
            return
        overlay = self.image.copy()
        # Zvýrazni vybranou komponentu
        if self.selected_component is not None and self.selected_component > 0:
            mask = (self.component_labels == self.selected_component)
            overlay = self.apply_overlay(overlay, mask, self.highlight_color)
        # Překresli hranice superpixelů (červeně)
        edge_mask = self.superpixel_edges if self.superpixel_edges is not None else np.zeros(self.image.shape[:2], dtype=bool)
        overlay[edge_mask] = [255, 0, 0]
        # Překresli ruční hranice (modře)
        for path in self.manual_borders:
            if len(path) > 1:
                for i in range(len(path) - 1):
                    rr, cc = draw.line(path[i][1], path[i][0], path[i+1][1], path[i+1][0])
                    overlay[rr, cc] = [0, 0, 255]
        # Převod na QImage
        if overlay.dtype != np.uint8:
            overlay = (255 * (overlay / overlay.max())).astype(np.uint8)
        h, w, ch = overlay.shape
        qimg = QImage(overlay.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def apply_overlay(self, img, mask, color):
        img = img.copy()
        alpha = color.alpha() / 255.0
        rgb = np.array([color.red(), color.green(), color.blue()])
        img[mask] = (1 - alpha) * img[mask] + alpha * rgb
        return img

    def image_mouse_move(self, event):
        if self.manual_mode and event.buttons() & Qt.LeftButton:
            pos = event.pos()
            label_w = self.image_label.width()
            label_h = self.image_label.height()
            pixmap = self.image_label.pixmap()
            if pixmap is None:
                return
            pm_w = pixmap.width()
            pm_h = pixmap.height()
            x = pos.x() - (label_w - pm_w) // 2
            y = pos.y() - (label_h - pm_h) // 2
            img_h, img_w = self.image.shape[:2]
            scale = min(label_w / img_w, label_h / img_h)
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            if 0 <= orig_x < img_w and 0 <= orig_y < img_h:
                self.current_border.append((orig_x, orig_y))
                self.display_image()

    def image_mouse_release(self, event):
        if self.manual_mode and event.button() == Qt.LeftButton and len(self.current_border) > 1:
            self.manual_borders.append(self.current_border[:])
            self.current_border = []
            self.recompute_components()
            self.display_image()
            self.update_border_panel()

    def image_clicked(self, event):
        if self.manual_mode:
            if event.button() == Qt.LeftButton:
                pos = event.pos()
                label_w = self.image_label.width()
                label_h = self.image_label.height()
                pixmap = self.image_label.pixmap()
                if pixmap is None:
                    return
                pm_w = pixmap.width()
                pm_h = pixmap.height()
                x = pos.x() - (label_w - pm_w) // 2
                y = pos.y() - (label_h - pm_h) // 2
                img_h, img_w = self.image.shape[:2]
                scale = min(label_w / img_w, label_h / img_h)
                orig_x = int(x / scale)
                orig_y = int(y / scale)
                if 0 <= orig_x < img_w and 0 <= orig_y < img_h:
                    self.current_border.append((orig_x, orig_y))
                    self.display_image()
            return
        # Výběr komponenty podle aktuální masky
        if self.image is None or self.component_labels is None:
            return
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
        img_h, img_w = self.image.shape[:2]
        scale = min(label_w / img_w, label_h / img_h)
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        if not (0 <= orig_x < img_w and 0 <= orig_y < img_h):
            return
        comp = self.component_labels[orig_y, orig_x]
        if comp == 0:
            return
        if self.selected_component == comp:
            self.selected_component = None
        else:
            self.selected_component = comp
        self.display_image()

    def update_border_panel(self):
        while self.border_layout.count():
            child = self.border_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        for i, path in enumerate(self.manual_borders):
            label = QLabel(f'Ruční čára {i+1}')
            remove_btn = QPushButton('Smazat')
            remove_btn.clicked.connect(lambda _, idx=i: self.remove_manual_border(idx))
            row = QHBoxLayout()
            row.addWidget(label)
            row.addWidget(remove_btn)
            row_widget = QWidget()
            row_widget.setLayout(row)
            self.border_layout.addWidget(row_widget)
        self.border_layout.addStretch(1)

    def remove_manual_border(self, idx):
        if 0 <= idx < len(self.manual_borders):
            self.manual_borders.pop(idx)
            self.recompute_components()
            self.display_image()
            self.update_border_panel()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SuperpixelAnnotator()
    window.show()
    sys.exit(app.exec_()) 