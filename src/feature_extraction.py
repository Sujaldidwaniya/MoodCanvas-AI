import cv2
import numpy as np
import os
from config import IMG_SIZE, CLASS_NAMES

def extract_visual_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    contrast = np.std(gray)
    mean_b, mean_g, mean_r = cv2.mean(img)[:3]

    hist_b = cv2.calcHist([img], [0], None, [8], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [8], [0, 256]).flatten()
    hist_r = cv2.calcHist([img], [2], None, [8], [0, 256]).flatten()
    hist = np.concatenate([hist_b, hist_g, hist_r])
    hist = hist / (hist.sum() + 1e-6)

    return np.concatenate([[brightness, contrast, mean_b, mean_g, mean_r], hist])