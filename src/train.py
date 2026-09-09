import numpy as np
import cv2
import os
from tensorflow.keras.utils import to_categorical

from config import IMG_SIZE, CLASS_NAMES, TRAIN_DIR, TEST_DIR, EPOCHS, BATCH_SIZE, MODEL_SAVE_PATH
from feature_extraction import extract_visual_features
from model import build_hybrid_model

def load_all_data(root_dir):
    images, features, labels = [], [], []
    for idx, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(root_dir, class_name)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0
            feat = extract_visual_features(fpath)
            images.append(img)
            features.append(feat)
            labels.append(idx)
    return np.array(images), np.array(features), to_categorical(labels, num_classes=len(CLASS_NAMES))

if __name__ == "__main__":
    print("Loading training data...")
    X_img_train, X_feat_train, y_train = load_all_data(TRAIN_DIR)
    print("Loading test data...")
    X_img_test, X_feat_test, y_test = load_all_data(TEST_DIR)

    model = build_hybrid_model(feature_dim=X_feat_train.shape[1])
    model.summary()

    model.fit(
        x=[X_img_train, X_feat_train],
        y=y_train,
        validation_data=([X_img_test, X_feat_test], y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    model.save(MODEL_SAVE_PATH)
    print("Model saved to", MODEL_SAVE_PATH)