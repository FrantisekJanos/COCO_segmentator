# Superpixel Segmentation Fakin-Annotator (PyQt5)

This is a fast, user-friendly desktop tool for creating segmentation annotations for machine learning in the COCO JSON format. The app is designed for efficient manual and semi-automatic annotation of objects in images using superpixel segmentation and interactive editing.

## Features

- **Superpixel segmentation**: Quickly split images into meaningful regions (superpixels) using SLIC.
- **Manual border drawing**: Draw custom borders to split or refine regions interactively.
- **Multi-region selection**: Select multiple regions at once to annotate complex objects.
- **Label management**: Assign labels to regions, reuse existing labels with one click, and manage all segmentations in a panel.
- **COCO JSON export**: Save all your segmentations in a single, valid COCO JSON file for ML training.
- **Preview JSON**: View the generated COCO JSON in a formatted, readable window before export.
- **Keyboard shortcuts**: Toggle manual border drawing mode with the 'C' key.

## How to Use

1. **Install dependencies**
   - Python 3.8+
   - Install required packages:
     ```
     pip install PyQt5 scikit-image numpy
     ```
2. **Run the app**
   ```
   python main_2.py
   ```
3. **Workflow**
   - Click "Load Image" and select an image file.
   - Adjust the number of superpixels using the slider or number box.
   - Select regions by clicking on them (multi-select is supported).
   - Optionally, draw custom borders in "Manual border drawing" mode (toggle with the checkbox or 'C' key).
   - Enter a label for the selected region(s) or use an existing label button.
   - Click "Add segmentation" to save the current selection and label.
   - Manage all segmentations in the right panel (delete, review, etc.).
   - When done, click "Export all to COCO JSON" to save all segmentations to a file.
   - Use "Show JSON" to preview the current COCO JSON in a readable format.

## Tips

- You can quickly reuse labels by clicking on the dynamically generated label buttons.
- You cannot add a segmentation without selecting at least one region and entering a label.
- All segmentations are kept in memory until you export them.

## License

MIT
