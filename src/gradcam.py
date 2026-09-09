import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import IMG_SIZE, MODEL_SAVE_PATH
from feature_extraction import extract_visual_features

def make_gradcam_heatmap(img_array, feat_array, model, last_conv_layer_name="last_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array, feat_array])
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img_path, output_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img_norm = img / 255.0
    img_array = np.expand_dims(img_norm, axis=0)
    feat_array = np.expand_dims(extract_visual_features(img_path), axis=0)

    heatmap = make_gradcam_heatmap(img_array, feat_array, model)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, superimposed)
    print("Saved:", output_path)

if __name__ == "__main__":
    model = load_model(MODEL_SAVE_PATH)
    # Example: change this to any real image path from your test set
    overlay_gradcam(r"C:\Users\sujal\OneDrive\Desktop\Final Projects\Moodcanvas\Dataset\test\angry\PrivateTest_88305.jpg", "../outputs/gradcam_result.jpg", model)