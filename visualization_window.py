import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QSizePolicy, QMessageBox, QFrame
)
from PyQt5.QtGui import QPixmap, QColor
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

def get_label_color(label):
    # Jednotná barva pro každý label (hash + HSV)
    h = abs(hash(label)) % 360
    return QColor.fromHsv(h, 200, 255)

class VisualizationWindow(QMainWindow):
    def __init__(self, segmentations_by_image, parent=None, on_delete_segmentation=None):
        super().__init__(parent)
        self.setWindowTitle("Vizualizace segmentací")
        self.resize(1200, 800)
        self.segmentations_by_image = segmentations_by_image  # dict: {image_path: [SegmentationEntry, ...]}
        self.image_paths = list(segmentations_by_image.keys())
        self.current_image = None
        self.current_segmentations = None
        self.current_np_image = None
        self.selected_segmentation_idx = None
        self.on_delete_segmentation = on_delete_segmentation

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

        # Panel se segmentacemi
        self.seg_panel = QWidget()
        self.seg_layout = QVBoxLayout()
        self.seg_panel.setLayout(self.seg_layout)
        self.seg_panel.setMinimumWidth(260)
        self.seg_panel.setMaximumWidth(400)
        self.seg_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(reload_btn)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget, 2)
        main_layout.addWidget(self.image_label, 5)
        main_layout.addWidget(self.seg_panel, 3)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def on_image_selected(self, row):
        if row < 0 or row >= len(self.image_paths):
            self.image_label.setText("Vyberte obrázek vlevo")
            self.seg_layout.setParent(None)
            self.selected_segmentation_idx = None
            return
        image_path = self.image_paths[row]
        self.current_image = image_path
        self.current_segmentations = self.segmentations_by_image[image_path]
        self.selected_segmentation_idx = None
        try:
            np_img = self.load_image(image_path)
            self.current_np_image = np_img
            self.update_seg_panel()
            self.update_overlay()
        except Exception as e:
            print("Chyba při načítání obrázku:")
            traceback.print_exc()
            self.image_label.setText(f"Nelze načíst obrázek: {image_path}\n{e}")

    def update_overlay(self):
        print("\n=== DEBUG: Začátek update_overlay ===")
        try:
            if self.current_np_image is not None and self.current_segmentations is not None:
                print("Připravuji overlay...")
                segs = [segmentation_entry_to_dict(e) for e in self.current_segmentations]
                print(f"Počet segmentací pro overlay: {len(segs)}")
                
                for i, s in enumerate(segs):
                    s['color'] = get_label_color(s['label'])
                    if i == self.selected_segmentation_idx:
                        s['outline_width'] = 8
                    else:
                        s['outline_width'] = 4
                    
                print("Vykresluji overlay...")
                pixmap = draw_segmentation_overlay(self.current_np_image, segs)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                print("Overlay vykreslen")
            else:
                print("Není co vykreslovat - chybí obrázek nebo segmentace")
            
        except Exception as e:
            print(f"\n!!! CHYBA v update_overlay !!!")
            print(f"Typ chyby: {type(e).__name__}")
            print(f"Chybová zpráva: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
        
        print("=== DEBUG: Konec update_overlay ===\n")

    def resizeEvent(self, event):
        self.update_overlay()
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
                self.update_overlay()
            except Exception as e:
                print("Chyba při načítání obrázku:")
                traceback.print_exc()
                QMessageBox.critical(self, "Chyba", f"Nelze načíst obrázek: {fname}\n{e}")

    def update_seg_panel(self):
        print("\n=== DEBUG: Začátek update_seg_panel ===")
        try:
            print("Mažu staré widgety...")
            while self.seg_layout.count():
                child = self.seg_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            
            if not self.current_segmentations:
                print("Žádné segmentace k zobrazení")
                return
            
            print(f"Vytvářím tlačítka pro {len(self.current_segmentations)} segmentací")
            for idx, seg in enumerate(self.current_segmentations):
                print(f"Vytvářím tlačítko pro segmentaci {idx}: {seg.label}")
                color = get_label_color(seg.label)
                row = QHBoxLayout()
                
                color_frame = QFrame()
                color_frame.setFixedSize(18, 18)
                color_frame.setStyleSheet(f"background: {color.name()}; border: 2px solid {color.name()}; border-radius: 4px;")
                row.addWidget(color_frame)
                
                label_btn = QPushButton(seg.label)
                label_btn.setStyleSheet(self._label_btn_style(idx, color))
                label_btn.clicked.connect(lambda _, i=idx: self.select_segmentation(i))
                row.addWidget(label_btn)
                
                remove_btn = QPushButton('Smazat')
                remove_btn.setStyleSheet("padding: 2px 8px;")
                remove_btn.clicked.connect(lambda _, i=idx: self.remove_segmentation(i))
                row.addWidget(remove_btn)
                
                row_widget = QWidget()
                row_widget.setLayout(row)
                self.seg_layout.addWidget(row_widget)
            
            print("Přidávám stretch...")
            self.seg_layout.addStretch(1)
            print("=== DEBUG: Konec update_seg_panel ===\n")
            
        except Exception as e:
            print(f"\n!!! CHYBA v update_seg_panel !!!")
            print(f"Typ chyby: {type(e).__name__}")
            print(f"Chybová zpráva: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()

    def _label_btn_style(self, idx, color):
        if idx == self.selected_segmentation_idx:
            return f"font-weight: bold; background: {color.lighter(170).name()}; padding: 2px 8px; border: 2px solid {color.name()}; border-radius: 6px;"
        else:
            return f"font-weight: normal; background: none; padding: 2px 8px; border: 2px solid {color.name()}; border-radius: 6px;"

    def select_segmentation(self, idx):
        self.selected_segmentation_idx = idx
        self.update_seg_panel()
        self.update_overlay()

    def remove_segmentation(self, idx):
        if self.current_image and 0 <= idx < len(self.current_segmentations):
            try:
                seg = self.current_segmentations[idx]
                seg_id = getattr(seg, 'id', None)
                
                # Odstraníme segmentaci z lokálního seznamu
                self.current_segmentations = self.current_segmentations[:idx] + self.current_segmentations[idx+1:]
                self.selected_segmentation_idx = None
                
                # Aktualizujeme UI
                self.update_seg_panel()
                self.update_overlay()
                
                # Až po úspěšné aktualizaci UI zavoláme callback
                if self.on_delete_segmentation and seg_id is not None:
                    self.on_delete_segmentation(self.current_image, seg_id)
                
            except Exception as e:
                print(f"Chyba při mazání segmentace: {e}")
                if self.current_image in self.segmentations_by_image:
                    self.current_segmentations = self.segmentations_by_image[self.current_image]
                    self.update_seg_panel()
                    self.update_overlay()

    def keyPressEvent(self, event):
        if self.current_segmentations and self.selected_segmentation_idx is not None:
            if event.key() == Qt.Key_Up:
                if self.selected_segmentation_idx > 0:
                    self.selected_segmentation_idx -= 1
                    self.update_seg_panel()
                    self.update_overlay()
                event.accept()
                return
            elif event.key() == Qt.Key_Down:
                if self.selected_segmentation_idx < len(self.current_segmentations) - 1:
                    self.selected_segmentation_idx += 1
                    self.update_seg_panel()
                    self.update_overlay()
                event.accept()
                return
            elif event.key() == Qt.Key_Delete:
                self.remove_segmentation(self.selected_segmentation_idx)
                event.accept()
                return
        super().keyPressEvent(event)

    def refresh_segmentations(self, segmentations_by_image):
        # Aktualizujeme pouze data, UI necháme být
        self.segmentations_by_image = segmentations_by_image
        self.image_paths = list(segmentations_by_image.keys())
        
        if self.current_image in self.image_paths:
            self.current_segmentations = self.segmentations_by_image[self.current_image]
            if self.current_segmentations is None:
                self.current_segmentations = []