import os
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Add src/ folder to Python path so we can import feature_extraction.py
SRC_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.append(SRC_PATH)
from feature_extraction import extract_visual_features

IMG_SIZE = (128, 128)
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'moodcanvas_model.keras')

print("Loading MoodCanvas model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img_norm = img / 255.0
    img_array = np.expand_dims(img_norm, axis=0)

    feat = extract_visual_features(img_path)
    feat_array = np.expand_dims(feat, axis=0)

    preds = model.predict([img_array, feat_array])[0]
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = round(float(np.max(preds)) * 100, 2)

    all_scores = {CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASS_NAMES))}
    return predicted_class, confidence, all_scores