import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QSizePolicy, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import numpy as np
from visualization_logic import draw_segmentation_overlay
import traceback


def segmentation_entry_to_dict(entry):
    # Pomocná funkce pro převod SegmentationEntry na dict kompatibilní s vizualizací
    return {
        'mask': entry.mask,
        'label': entry.label,
        'color': entry.color,
        'polygon': entry.polygon
    }

class VisualizationWindow(QMainWindow):
    def __init__(self, segmentations_by_image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vizualizace segmentací")
        self.resize(1000, 700)
        self.segmentations_by_image = segmentations_by_image  # dict: {image_path: [SegmentationEntry, ...]}
        self.image_paths = list(segmentations_by_image.keys())
        self.current_image = None
        self.current_segmentations = None
        self.current_np_image = None

        # UI
        self.list_widget = QListWidget()
        for path in self.image_paths:
            self.list_widget.addItem(path)
        self.list_widget.currentRowChanged.connect(self.on_image_selected)

        self.image_label = QLabel("Vyberte obrázek vlevo")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        reload_btn = QPushButton("Načíst obrázek ze souboru")
        reload_btn.clicked.connect(self.load_image_from_disk)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(reload_btn)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget, 2)
        main_layout.addWidget(self.image_label, 5)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def on_image_selected(self, row):
        if row < 0 or row >= len(self.image_paths):
            self.image_label.setText("Vyberte obrázek vlevo")
            return
        image_path = self.image_paths[row]
        self.current_image = image_path
        self.current_segmentations = self.segmentations_by_image[image_path]
        # Zkus načíst obrázek
        try:
            np_img = self.load_image(image_path)
            self.current_np_image = np_img
            # Převod segmentací na dicty
            segs = [segmentation_entry_to_dict(e) for e in self.current_segmentations]
            try:
                pixmap = draw_segmentation_overlay(np_img, segs)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except Exception as e:
                print("Chyba při vykreslování overlay:")
                traceback.print_exc()
                self.image_label.setText(f"Chyba při vykreslování segmentací:\n{e}")
        except Exception as e:
            print("Chyba při načítání obrázku:")
            traceback.print_exc()
            self.image_label.setText(f"Nelze načíst obrázek: {image_path}\n{e}")

    def resizeEvent(self, event):
        # Při změně velikosti okna přepočítat zobrazení obrázku
        if self.current_np_image is not None and self.current_segmentations is not None:
            try:
                segs = [segmentation_entry_to_dict(e) for e in self.current_segmentations]
                pixmap = draw_segmentation_overlay(self.current_np_image, segs)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except Exception as e:
                print("Chyba při vykreslování overlay:")
                traceback.print_exc()
                self.image_label.setText(f"Chyba při vykreslování segmentací:\n{e}")
        super().resizeEvent(event)

    def load_image(self, path):
        # Načte obrázek jako numpy array (RGB)
        from skimage.io import imread
        img = imread(path)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[..., :3]
        return img

    def load_image_from_disk(self):
        if not self.current_image:
            QMessageBox.warning(self, "Varování", "Nejprve vyberte obrázek v seznamu.")
            return
        fname, _ = QFileDialog.getOpenFileName(self, "Vyberte obrázek", "", "Obrázky (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            try:
                np_img = self.load_image(fname)
                self.current_np_image = np_img
                segs = [segmentation_entry_to_dict(e) for e in self.current_segmentations]
                try:
                    pixmap = draw_segmentation_overlay(np_img, segs)
                    self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                except Exception as e:
                    print("Chyba při vykreslování overlay:")
                    traceback.print_exc()
                    self.image_label.setText(f"Chyba při vykreslování segmentací:\n{e}")
            except Exception as e:
                print("Chyba při načítání obrázku:")
                traceback.print_exc()
                QMessageBox.critical(self, "Chyba", f"Nelze načíst obrázek: {fname}\n{e}") 