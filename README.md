#  MoodCanvas AI — Artwork Mood Classification

Deep learning system that looks at a painting, doodle, or photo and classifies its visual mood — **Happy, Calm, Sad, Angry,** or **Neutral** — by combining a CNN's learned image features with handcrafted visual features (brightness, contrast, color distribution) through a hybrid ANN classifier.

[**📂 GitHub**](#)

`Python` `TensorFlow / Keras` `CNN` `ANN` `OpenCV` `Grad-CAM` `Django`

---

## Overview

MoodCanvas AI predicts the emotional tone of an image rather than the emotion of a face — it works on artwork, illustrations, and photos alike. It does this with a **hybrid architecture**: a convolutional branch learns spatial patterns directly from pixels, while a parallel ANN branch reasons over engineered visual features like brightness, contrast, and RGB color histograms. The two branches are merged before the final classification layer, so the model gets both "what's in the image" and "how the image feels" as signals.

Grad-CAM is used on top of the trained model to visualize *which regions of the image* drove a given prediction, making the classifier's decisions interpretable rather than a black box.

## Features

- **Hybrid CNN + ANN pipeline** — image features from a CNN merged with handcrafted visual features (brightness, contrast, color distribution) through a dense ANN classifier
- **Image preprocessing & augmentation** to improve generalization across varied artwork and photo styles
- **Grad-CAM interpretability** — heatmap overlays showing which regions of an image influenced the prediction
- **Web demo built with Django** — upload an image, get the predicted mood, confidence score, and full class-probability breakdown
- **Evaluation suite** — classification report and confusion matrix for measuring per-class performance

## Demo

<table>
<tr>
<td align="center">
<img src="https://github.com/user-attachments/assets/fd1490cc-cab2-4214-acdc-251b8ddb2e82" width="300">
<br>
<b>Grad-CAM Visualization</b>
</td>

<td align="center">
<img src="https://github.com/user-attachments/assets/ddd6f821-8926-4445-a7e3-8c12748cf950" width="300">
<br>
<b>Confusion Matrix</b>
</td>
</tr>
</table>
## Model Architecture

```
                 ┌────────────────────┐
Image (128x128) →│  CNN (Conv+Pool x3) │→ GAP → Dense(128) → Dropout
                 └────────────────────┘                        │
                                                                 ├─→ Concatenate → Dense(64) → Dropout → Softmax(5)
                 ┌────────────────────┐                        │
Visual features →│  ANN (Dense x2)     │→ Dropout ──────────────┘
(brightness,     └────────────────────┘
 contrast, RGB
 histograms)
```

- **CNN branch:** 3 convolutional blocks (32 → 64 → 128 filters) with max pooling, followed by global average pooling and a dense layer
- **ANN branch:** Dense layers over an engineered feature vector — mean brightness, contrast (std. dev.), mean RGB values, and 8-bin color histograms per channel (extracted with OpenCV)
- **Fusion:** Both branches are concatenated and passed through a final dense classification head
- **Interpretability:** Grad-CAM is computed on the CNN branch's last convolutional layer to highlight the regions that influenced each prediction

## Tech Stack

| Category | Tools |
|---|---|
| Modeling | TensorFlow / Keras |
| Image processing | OpenCV |
| Interpretability | Grad-CAM |
| Evaluation | scikit-learn, seaborn, matplotlib |
| Web app | Django |
| Language | Python |

## Project Structure

```
MoodCanvas/
├── src/
│   ├── config.py              # Paths, image size, class names, hyperparameters
│   ├── feature_extraction.py  # Brightness, contrast, color histogram extraction (OpenCV)
│   ├── model.py                # Hybrid CNN + ANN architecture
│   ├── train.py                 # Data loading + training loop
│   ├── evaluate.py              # Classification report + confusion matrix
│   └── gradcam.py               # Grad-CAM heatmap generation
├── models/
│   └── moodcanvas_model.keras   # Trained model weights
├── outputs/
│   ├── confusion_matrix.png
│   └── gradcam_result.jpg
└── webapp/                      # Django web app
    ├── manage.py
    └── classifier/
        ├── views.py              # Handles image upload + prediction
        ├── ml_model.py           # Loads model, runs inference
        ├── urls.py
        └── templates/classifier/index.html
```

## Setup & Installation

```bash
git clone https://github.com/<your-username>/moodcanvas-ai.git
cd moodcanvas-ai

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**`requirements.txt`**

```
tensorflow
opencv-python
numpy
scikit-learn
seaborn
matplotlib
django
```

## Dataset

Organize your dataset into class-labeled folders before training:

```
Dataset/
├── train/
│   ├── happy/
│   ├── calm/
│   ├── sad/
│   ├── angry/
│   └── neutral/
└── test/
    ├── happy/
    ├── calm/
    ├── sad/
    ├── angry/
    └── neutral/
```

Update `TRAIN_DIR`, `TEST_DIR`, and `MODEL_SAVE_PATH` in `src/config.py` to point to your local dataset and desired model output path.

## Training

```bash
cd src
python train.py
```

This loads every image, extracts its visual features, trains the hybrid CNN + ANN model, and saves the trained model to the path set in `config.py`.

## Evaluation

```bash
cd src
python evaluate.py
```

Generates a classification report in the console and saves a confusion matrix to `outputs/confusion_matrix.png`.

## Grad-CAM Visualization

```bash
cd src
python gradcam.py
```

Runs Grad-CAM on a sample image and saves the heatmap overlay to `outputs/gradcam_result.jpg`, highlighting the regions that most influenced the model's prediction.

## Running the Web App

```bash
cd webapp
python manage.py runserver
```

Visit `http://127.0.0.1:8000/`, upload an image, and view the predicted mood, confidence score, and per-class probability breakdown.

## Results

| Metric | Score |
|---|---|
| Validation Accuracy | 54%|
| Macro F1-score | 50% |

See `outputs/confusion_matrix.png` for the full per-class breakdown.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Sujal Didwaniya** — [GitHub](#) · [LinkedIn](#)
