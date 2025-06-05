from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SegmentationEntry:
    image_path: str
    label: str
    mask: np.ndarray = None  # 2D bool/uint8 mask
    polygon: Optional[List] = None  # list of (x, y)
    color: Optional[tuple] = None  # (R, G, B)
    # další metadata lze přidat dle potřeby

class SegmentationStorage:
    def __init__(self):
        # {image_path: [SegmentationEntry, ...]}
        self.segmentations_by_image: Dict[str, List[SegmentationEntry]] = {}

    def add_segmentation(self, entry: SegmentationEntry):
        if entry.image_path not in self.segmentations_by_image:
            self.segmentations_by_image[entry.image_path] = []
        self.segmentations_by_image[entry.image_path].append(entry)

    def remove_segmentation(self, image_path: str, idx: int):
        if image_path in self.segmentations_by_image:
            if 0 <= idx < len(self.segmentations_by_image[image_path]):
                del self.segmentations_by_image[image_path][idx]
                if not self.segmentations_by_image[image_path]:
                    del self.segmentations_by_image[image_path]

    def get_segmentations(self, image_path: str) -> List[SegmentationEntry]:
        return self.segmentations_by_image.get(image_path, [])

    def get_all_segmentations(self) -> Dict[str, List[SegmentationEntry]]:
        return self.segmentations_by_image

    def clear(self):
        self.segmentations_by_image.clear() 