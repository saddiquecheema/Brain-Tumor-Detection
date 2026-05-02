import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import cv2
import json
import base64
import io
import os

app = Flask(__name__)

# Model aur class indices load karo

# Keras 3.x / TF 2.x compatibility fix for Dense layer
try:
    original_from_config = tf.keras.layers.Dense.from_config
    def patched_from_config(cls, config):
        if 'quantization_config' in config:
            del config['quantization_config']
        return original_from_config(config)
    tf.keras.layers.Dense.from_config = classmethod(patched_from_config)
except Exception:
    pass

# .keras format support — pehle .keras try karo, phir .h5 fallback
if os.path.exists('brain_tumor_model.keras'):
    model = tf.keras.models.load_model('brain_tumor_model.keras')
    pass
elif os.path.exists('brain_tumor_model.h5'):
    model = tf.keras.models.load_model('brain_tumor_model.h5')
    pass
else:
    raise FileNotFoundError(
        "Model file nahi mila!\n"
        "   brain_tumor_model.keras ya brain_tumor_model.h5 chahiye.\n"
        "   Is file ko app.py ke saath same folder mein rakhen."
    )

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

descriptions = {
    'glioma':     'Brain tissue ke andar tumor hai. Fori tor par doctor se rabita karen.',
    'meningioma': 'Brain ki covering par tumor hai. Surgery se theek ho sakta hai.',
    'notumor':    'MRI bilkul normal hai. Koi tumor nahi mila.',
    'pituitary':  'Hormone gland par tumor hai. Treatment available hai.'
}

advice = {
    'glioma':     'Turant neurologist se mile. MRI ke sath report le jayen.',
    'meningioma': 'Neurosurgeon se consult karen. Aksar operable hota hai.',
    'notumor':    'Koi action ki zaroorat nahi. Regular checkup jari rakhen.',
    'pituitary':  'Endocrinologist se mile. Hormone test karwayen.'
}


def annotate_image(orig_img, class_name, confidence):
    h, w   = orig_img.shape[:2]
    output = orig_img.copy()

    if class_name == 'notumor':
        return output

    gray    = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    top_thresh  = np.percentile(blurred, 97)
    bright_mask = np.uint8(blurred >= top_thresh) * 255

    kernel  = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN,  kernel)
    cleaned = cv2.morphologyEx(cleaned,     cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_area = h * w * 0.15
        min_area = h * w * 0.005
        valid    = [c for c in contours
                    if min_area < cv2.contourArea(c) < max_area]
        if not valid:
            valid = [c for c in contours if cv2.contourArea(c) < h * w * 0.20]
        if not valid:
            valid = contours

        largest      = max(valid, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)

        pad = 5
        x   = max(0, x - pad)
        y   = max(0, y - pad)
        bw  = min(w - x, bw + pad * 2)
        bh  = min(h - y, bh + pad * 2)

        box_color = {
            'glioma':     (255, 80,  80),
            'meningioma': (80,  255, 80),
            'pituitary':  (80,  150, 255)
        }.get(class_name, (255, 255, 0))

        cv2.rectangle(output, (x, y), (x+bw, y+bh), box_color, 2)

        label      = f'{class_name} {confidence:.0f}%'
        font_scale = 0.45
        thickness  = 1
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        lx = x + 4
        ly = y + th + 4
        cv2.rectangle(output, (lx-2, y+2), (lx+tw+2, y+th+6), box_color, -1)
        cv2.putText(output, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return output


def img_to_base64(img_array):
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    buf     = io.BytesIO()
    img_pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Koi file nahi mili'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File select nahi ki'}), 400

    try:
        img_bytes = file.read()
        img_pil   = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        orig_img  = np.array(img_pil)

        # Prediction — MobileNetV2 preprocessing
        img_resized = img_pil.resize((224, 224))
        img_array   = np.expand_dims(np.array(img_resized), axis=0)
        img_array   = preprocess_input(img_array.astype(np.float32))
        preds       = model.predict(img_array, verbose=0)

        class_idx  = int(np.argmax(preds[0]))
        class_name = index_to_class[class_idx]
        confidence = float(preds[0][class_idx] * 100)

        all_probs = {
            index_to_class[i]: round(float(preds[0][i] * 100), 2)
            for i in range(4)
        }

        # Annotated image
        annotated     = annotate_image(orig_img, class_name, confidence)
        annotated_b64 = img_to_base64(annotated)
        original_b64  = img_to_base64(orig_img)

        return jsonify({
            'class':       class_name,
            'confidence':  round(confidence, 2),
            'description': descriptions[class_name],
            'advice':      advice[class_name],
            'all_probs':   all_probs,
            'original':    original_b64,
            'annotated':   annotated_b64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
