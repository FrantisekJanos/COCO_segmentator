import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QFrame, QColorDialog, QScrollArea, QLineEdit, QCheckBox, QTextEdit, QDialog, QVBoxLayout as QVLayout, QDialogButtonBox, QHBoxLayout as QHLayout, QSpinBox
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt
from skimage import io, color, segmentation, img_as_ubyte, draw
from scipy.ndimage import label as ndi_label
import json
from segmentation_storage import SegmentationStorage, SegmentationEntry
from visualization_window import VisualizationWindow

class SuperpixelAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Superpixel Segmentační fakin-annotátor (PyQt5)')
        self.setGeometry(100, 100, 1200, 800)
        self.image = None
        self.superpixel_edges = None
        self.manual_borders = []  # seznam ručně dokreslených hranic (každá je seznam bodů)
        self.current_border = []
        self.component_labels = None  # maska komponent
        self.selected_components = set()  # multi-select
        self.highlight_color = QColor(0, 255, 0, 120)
        self.manual_mode = False
        self.label_edit = None
        self.current_label = ''
        self._fixed_pixmap_size = None  # pro fixní velikost obrázku
        self.last_image_path = None
        self.segmentations = []  # seznam segmentací v paměti
        self.label_buttons_widget = None
        self.label_buttons_layout = None
        self.seg_storage = SegmentationStorage()
        self.vis_window = None
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
        # Spinbox pro počet superpixelů
        self.slic_spin = QSpinBox()
        self.slic_spin.setMinimum(20)
        self.slic_spin.setMaximum(500)
        self.slic_spin.setValue(100)
        self.slic_spin.valueChanged.connect(self.slic_slider.setValue)
        self.slic_slider.valueChanged.connect(self.slic_spin.setValue)
        slic_layout = QHBoxLayout()
        slic_layout.addWidget(self.slic_label)
        slic_layout.addWidget(self.slic_slider)
        slic_layout.addWidget(self.slic_spin)

        # Barva zvýraznění
        self.color_btn = QPushButton('Barva zvýraznění')
        self.color_btn.setStyleSheet(f'background-color: {self.highlight_color.name()};')
        self.color_btn.clicked.connect(self.choose_color)

        # Režim ručního kreslení
        self.manual_checkbox = QCheckBox('Ruční dělení oblasti čárou (nebo klávesa C)')
        self.manual_checkbox.stateChanged.connect(self.toggle_manual_mode)

        # Label pro vybrané oblasti
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText('Label pro vybrané oblasti')
        self.label_edit.textChanged.connect(self.set_label)

        # Dynamická tlačítka labelů
        self.label_buttons_widget = QWidget()
        self.label_buttons_layout = QHLayout()
        self.label_buttons_widget.setLayout(self.label_buttons_layout)
        self.update_label_buttons()

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

        # Tlačítka pro COCO JSON
        self.save_json_btn = QPushButton('Přidat segmentaci')
        self.save_json_btn.clicked.connect(lambda: self.save_coco_json())
        self.show_json_btn = QPushButton('Zobrazit JSON')
        self.show_json_btn.clicked.connect(self.show_coco_json)
        self.export_all_btn = QPushButton('Exportovat vše do COCO JSON')
        self.export_all_btn.clicked.connect(self.export_all_coco_json)
        self.new_label_btn = QPushButton('Nový label')
        self.new_label_btn.clicked.connect(self.new_label)
        self.visualize_btn = QPushButton('Vizualizace segmentací')
        self.visualize_btn.clicked.connect(self.open_visualization)

        # Panel pro správu segmentací
        self.seg_panel = QWidget()
        self.seg_layout = QVBoxLayout()
        self.seg_panel.setLayout(self.seg_layout)
        self.seg_scroll = QScrollArea()
        self.seg_scroll.setWidgetResizable(True)
        self.seg_scroll.setWidget(self.seg_panel)
        self.seg_scroll.setMinimumWidth(220)
        self.seg_scroll.setMaximumWidth(350)
        self.seg_scroll.setWindowTitle('Segmentace')

        # Layouts
        slider_layout = QVBoxLayout()
        slider_layout.addLayout(slic_layout)
        slider_layout.addWidget(load_btn)
        slider_layout.addWidget(self.color_btn)
        slider_layout.addWidget(self.manual_checkbox)
        slider_layout.addWidget(self.label_edit)
        slider_layout.addWidget(self.label_buttons_widget)
        slider_layout.addWidget(self.save_json_btn)
        slider_layout.addWidget(self.new_label_btn)
        slider_layout.addWidget(self.show_json_btn)
        slider_layout.addWidget(self.export_all_btn)
        slider_layout.addWidget(self.visualize_btn)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addLayout(slider_layout)
        left_layout.addWidget(self.image_label)
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.border_scroll)
        main_layout.addWidget(self.seg_scroll)

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

    def set_label(self, text):
        self.current_label = text

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Vyberte obrázek', '', 'Obrázky (*.png *.jpg *.jpeg *.bmp)')
        if file_name:
            self.image = io.imread(file_name)
            self.last_image_path = file_name
            if self.image.ndim == 2:
                self.image = color.gray2rgb(self.image)
            self.manual_borders = []
            self.current_border = []
            self.selected_components = set()
            self.current_label = ''
            self.label_edit.setText('')
            self.update_superpixels()

    def update_superpixels(self):
        if self.image is None:
            return
        n_segments = self.slic_slider.value()
        self.slic_label.setText(f'Počet oblastí: {n_segments}')
        segments = segmentation.slic(self.image, n_segments=n_segments, compactness=10, start_label=1)
        edges = segmentation.find_boundaries(segments, mode='thick')
        self.superpixel_edges = edges
        self.selected_components = set()
        self.current_label = ''
        self.label_edit.setText('')
        self.recompute_components()
        self.display_image()
        self.update_border_panel()

    def recompute_components(self):
        if self.image is None:
            return
        border_mask = self.superpixel_edges.copy()
        for path in self.manual_borders:
            if len(path) > 1:
                for i in range(len(path) - 1):
                    # Tlustá čára (2px): vykresli okolní pixely
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            rr, cc = draw.line(path[i][1]+dy, path[i][0]+dx, path[i+1][1]+dy, path[i+1][0]+dx)
                            border_mask[rr, cc] = True
        area_mask = ~border_mask
        labels, _ = ndi_label(area_mask)
        self.component_labels = labels

    def display_image(self):
        if self.image is None or self.component_labels is None:
            return
        overlay = self.image.copy()
        # Zvýrazni všechny vybrané komponenty
        for comp in self.selected_components:
            if comp > 0:
                mask = (self.component_labels == comp)
                overlay = self.apply_overlay(overlay, mask, self.highlight_color)
        # Překresli hranice superpixelů (červeně, tloušťka 2)
        edge_mask = self.superpixel_edges if self.superpixel_edges is not None else np.zeros(self.image.shape[:2], dtype=bool)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                shifted = np.roll(edge_mask, shift=(dy, dx), axis=(0, 1))
                overlay[shifted] = [255, 0, 0]
        # Překresli ruční hranice (červeně, tloušťka 2)
        for path in self.manual_borders:
            if len(path) > 1:
                for i in range(len(path) - 1):
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            rr, cc = draw.line(path[i][1]+dy, path[i][0]+dx, path[i+1][1]+dy, path[i+1][0]+dx)
                            overlay[rr, cc] = [255, 0, 0]
        # Pokud právě kreslíme, vykresli aktuální čáru
        if self.manual_mode and len(self.current_border) > 1:
            for i in range(len(self.current_border) - 1):
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        rr, cc = draw.line(
                            self.current_border[i][1]+dy, self.current_border[i][0]+dx,
                            self.current_border[i+1][1]+dy, self.current_border[i+1][0]+dx)
                        overlay[rr, cc] = [255, 0, 0]
        if overlay.dtype != np.uint8:
            overlay = (255 * (overlay / overlay.max())).astype(np.uint8)
        h, w, ch = overlay.shape
        qimg = QImage(overlay.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # Oprava zvětšování: použij fixní velikost pixmapy
        if self._fixed_pixmap_size is None:
            self._fixed_pixmap_size = self.image_label.size()
        pixmap = pixmap.scaled(self._fixed_pixmap_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def apply_overlay(self, img, mask, color):
        img = img.copy()
        alpha = color.alpha() / 255.0
        rgb = np.array([color.red(), color.green(), color.blue()])
        img[mask] = (1 - alpha) * img[mask] + alpha * rgb
        return img

    def image_clicked(self, event):
        if self.manual_mode:
            if event.button() == Qt.LeftButton:
                pos = event.pos()
                label_w = self.image_label.width()
                label_h = self.image_label.height()
                pixmap = self.image_label.pixmap()
                if pixmap is None:
                    event.accept()
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
            event.accept()
            return
        # Multi-select komponent
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
        if comp in self.selected_components:
            self.selected_components.remove(comp)
        else:
            self.selected_components.add(comp)
        self.display_image()
        event.accept()

    def image_mouse_move(self, event):
        if self.manual_mode and event.buttons() & Qt.LeftButton:
            pos = event.pos()
            label_w = self.image_label.width()
            label_h = self.image_label.height()
            pixmap = self.image_label.pixmap()
            if pixmap is None:
                event.accept()
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
            event.accept()

    def image_mouse_release(self, event):
        if self.manual_mode and event.button() == Qt.LeftButton and len(self.current_border) > 1:
            self.manual_borders.append(self.current_border[:])
            self.current_border = []
            self.recompute_components()
            self.display_image()
            self.update_border_panel()
            event.accept()

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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_C:
            self.manual_checkbox.setChecked(not self.manual_checkbox.isChecked())
            event.accept()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        # Při změně velikosti okna aktualizuj fixní velikost pixmapy
        self._fixed_pixmap_size = self.image_label.size()
        self.display_image()
        super().resizeEvent(event)

    def update_label_buttons(self):
        # Smaž stará tlačítka
        while self.label_buttons_layout.count():
            child = self.label_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        # Najdi unikátní labely
        labels = list({ann['label'] for ann in self.segmentations})
        labels.sort()
        for label in labels:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, l=label: self.save_coco_json_with_label(l))
            self.label_buttons_layout.addWidget(btn)
        self.label_buttons_layout.addStretch(1)

    def save_coco_json_with_label(self, label):
        if not self.selected_components or self.image is None:
            return
        self.current_label = label
        coco_ann = self.create_coco_annotation(label_override=label)
        self.segmentations.append(coco_ann)
        self.update_seg_panel()
        self.update_label_buttons()
        self.new_label()  # automaticky připrav nový label
        # --- Uložit do storage ---
        self.save_to_storage(coco_ann)

    def save_coco_json(self):
        from PyQt5.QtWidgets import QMessageBox
        label = self.label_edit.text().strip()
        if not label:
            QMessageBox.warning(self, 'Chyba', 'Zadejte název labelu!')
            return
        if not self.selected_components or self.image is None:
            QMessageBox.warning(self, 'Chyba', 'Vyberte alespoň jednu oblast!')
            return
        self.current_label = label
        coco_ann = self.create_coco_annotation(label_override=label)
        self.segmentations.append(coco_ann)
        self.update_seg_panel()
        self.update_label_buttons()
        self.new_label()  # automaticky připrav nový label
        # --- Uložit do storage ---
        self.save_to_storage(coco_ann)

    def create_coco_annotation(self, label_override=None):
        h, w = self.image.shape[:2]
        annotation_id = len(self.segmentations) + 1
        category_id = 1
        mask = np.zeros((h, w), dtype=np.uint8)
        for comp in self.selected_components:
            mask = mask | (self.component_labels == comp)
        from skimage import measure
        segmentation = []
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            poly = []
            for y, x in contour:
                poly.extend([float(x), float(y)])
            if len(poly) >= 6:
                segmentation.append(poly)
        area = float(np.sum(mask))
        bbox = self.mask_to_bbox(mask)
        label = label_override if label_override is not None else (self.current_label if self.current_label else f'object_{annotation_id}')
        return {
            "id": annotation_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
            "label": label
        }

    def show_coco_json(self):
        if not self.segmentations:
            return
        coco = self.create_coco_json_all()
        dlg = QDialog(self)
        dlg.setWindowTitle('COCO JSON')
        layout = QVLayout()
        text = QTextEdit()
        text.setReadOnly(True)
        text.setText(json.dumps(coco, ensure_ascii=False, indent=2))
        layout.addWidget(text)
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        layout.addWidget(btns)
        dlg.setLayout(layout)
        dlg.resize(700, 600)
        dlg.exec_()

    def export_all_coco_json(self):
        if not self.segmentations:
            return
        coco = self.create_coco_json_all()
        file_name, _ = QFileDialog.getSaveFileName(self, 'Exportovat vše do COCO JSON', 'segmentace.json', 'JSON (*.json)')
        if file_name:
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(coco, f, ensure_ascii=False, indent=2)

    def create_coco_json_all(self):
        h, w = self.image.shape[:2]
        file_name = self.last_image_path.split('/')[-1] if self.last_image_path else 'image.png'
        images = [{
            "id": 1,
            "file_name": file_name,
            "width": w,
            "height": h
        }]
        annotations = []
        categories = []
        cat_map = {}
        for ann in self.segmentations:
            annotations.append({k: v for k, v in ann.items() if k != 'label'})
            if ann['label'] not in cat_map:
                cat_id = len(cat_map) + 1
                cat_map[ann['label']] = cat_id
                categories.append({"id": cat_id, "name": ann['label']})
            annotations[-1]['category_id'] = cat_map[ann['label']]
        return {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

    def update_seg_panel(self):
        # Vyčistí panel
        while self.seg_layout.count():
            child = self.seg_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        # Kontrola, zda jsou segmentace platné
        if not self.segmentations:
            return
        # Generování tlačítek na základě aktuálního seznamu segmentací
        for ann in self.segmentations:
            label = QLabel(f"{ann['label']} (oblastí: {len(ann['segmentation'])})")
            remove_btn = QPushButton('Smazat')
            remove_btn.clicked.connect(lambda _, seg_id=ann['id']: self.remove_segmentation_by_id(seg_id))
            row = QHBoxLayout()
            row.addWidget(label)
            row.addWidget(remove_btn)
            row_widget = QWidget()
            row_widget.setLayout(row)
            self.seg_layout.addWidget(row_widget)
        self.seg_layout.addStretch(1)

    def remove_segmentation(self, idx):
        # Tato metoda už nebude používaná pro mazání z UI, ale zachovám ji pro případné jiné použití
        if 0 <= idx < len(self.segmentations):
            seg_id = self.segmentations[idx]['id']
            self.remove_segmentation_by_id(seg_id)

    def remove_segmentation_by_id(self, seg_id):
        # Smaže segmentaci podle id z hlavního seznamu segmentací
        self.segmentations = [s for s in self.segmentations if s['id'] != seg_id]
        # --- Synchronizace s vizualizačním oknem ---
        if hasattr(self, 'vis_window') and self.vis_window is not None:
            self.vis_window.refresh_segmentations(self.seg_storage.get_all_segmentations())
        self.update_seg_panel()
        self.update_label_buttons()

    def mask_to_bbox(self, mask):
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, 0, 0]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

    def new_label(self):
        self.selected_components = set()
        self.current_label = ''
        self.label_edit.setText('')
        self.display_image()

    def save_to_storage(self, coco_ann):
        if self.last_image_path is None:
            return
        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        from skimage.draw import polygon as skpolygon
        for seg in coco_ann['segmentation']:
            if len(seg) >= 6:
                xs = seg[0::2]
                ys = seg[1::2]
                rr, cc = skpolygon(ys, xs, shape=mask.shape)
                mask[rr, cc] = 1
        entry = SegmentationEntry(
            id=coco_ann['id'],
            image_path=self.last_image_path,
            label=coco_ann['label'],
            mask=mask,
            polygon=None,
            color=None
        )
        self.seg_storage.add_segmentation(entry)

    def open_visualization(self):
        segs = self.seg_storage.get_all_segmentations()
        if not segs:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, 'Vizualizace', 'Nejsou k dispozici žádné segmentace.')
            return
        self.vis_window = VisualizationWindow(segs, on_delete_segmentation=self.on_delete_segmentation)
        self.vis_window.destroyed.connect(self.on_vis_window_closed)
        self.vis_window.show()

    def on_delete_segmentation(self, image_path, seg_id):
        # Smaže segmentaci v hlavní storage i v JSON
        self.seg_storage.remove_segmentation_by_id(image_path, seg_id)
        self.remove_segmentation_by_id(seg_id)
        # Kontrola, zda jsou objekty UI platné
        if hasattr(self, 'seg_panel') and self.seg_panel is not None:
            self.update_seg_panel()
        if hasattr(self, 'label_buttons_widget') and self.label_buttons_widget is not None:
            self.update_label_buttons()

    def on_vis_window_closed(self):
        self.vis_window = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SuperpixelAnnotator()
    window.show()
    sys.exit(app.exec_()) 