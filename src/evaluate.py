import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from config import CLASS_NAMES, MODEL_SAVE_PATH, TEST_DIR
from train import load_all_data

model = load_model(MODEL_SAVE_PATH)
X_img_test, X_feat_test, y_test = load_all_data(TEST_DIR)

preds = model.predict([X_img_test, X_feat_test])
y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("../outputs/confusion_matrix.png")
plt.show()