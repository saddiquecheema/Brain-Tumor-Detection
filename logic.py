import os
import json
import base64
import io
import numpy as np
import cv2
import requests
from PIL import Image
import config

# ── Model Loader ──────────────────────────────────────────────────────
def get_keras_loader():
    try:
        import keras
        return keras.models.load_model, "Keras 3"
    except ImportError:
        try:
            import tensorflow.keras as tfk
            return tfk.models.load_model, "tensorflow.keras"
        except ImportError:
            import tensorflow as tf
            return tf.keras.models.load_model, "tensorflow"

def build_brain_tumor_model():
    try:
        from keras import layers, models, applications
        base_model = applications.EfficientNetB0(
            weights=None, include_top=False, input_shape=(224, 224, 3)
        )
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(4, activation='softmax')
        ])
        return model
    except Exception as e:
        print(f"[ERROR] Model building failed: {e}")
        return None

def load_brain_tumor_model():
    keras_load, framework = get_keras_loader()
    print(f"[INFO] Using {framework} for model loading")
    
    model = None
    if os.path.exists(config.KERAS_PATH):
        try:
            model = keras_load(config.KERAS_PATH, compile=False)
            print(f"[INFO] Model loaded from: {os.path.basename(config.KERAS_PATH)}")
        except Exception as e:
            print(f"[WARN] Direct load failed: {e}. Trying weights fallback...")
            model = build_brain_tumor_model()
            if model is not None and os.path.exists(config.WEIGHTS_PATH):
                try:
                    model.load_weights(config.WEIGHTS_PATH)
                    print(f"[INFO] Model weights loaded from: {os.path.basename(config.WEIGHTS_PATH)}")
                except Exception as e2:
                    print(f"[ERROR] Weights load failed: {e2}")
                    model = None
    else:
        print("[WARN] .keras file not found. Trying weights only...")
        model = build_brain_tumor_model()
        if model is not None and os.path.exists(config.WEIGHTS_PATH):
            try:
                model.load_weights(config.WEIGHTS_PATH)
                print(f"[INFO] Model weights loaded from: {os.path.basename(config.WEIGHTS_PATH)}")
            except Exception as e:
                print(f"[ERROR] Weights load failed: {e}")
                model = None
    return model

def load_class_indices():
    class_indices = None
    for jp in config.JSON_CANDIDATES:
        if os.path.exists(jp):
            with open(jp, 'r') as f:
                class_indices = json.load(f)
            print(f"[INFO] Class indices loaded: {os.path.basename(jp)}")
            break
    
    if class_indices is None:
        class_indices = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
        print("[WARN] class_indices.json not found — using default")
    return class_indices

# ── Image Processing ──────────────────────────────────────────────────
def annotate_image(orig_img, class_name, confidence):
    h, w   = orig_img.shape[:2]
    output = orig_img.copy()

    if class_name == 'notumor':
        return output

    # 1. Brain Area Focus
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.30), int(h * 0.35))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    kernel_mask = np.ones((11, 11), np.uint8)
    mask = cv2.erode(mask, kernel_mask, iterations=2)

    gray = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
    masked_gray = cv2.bitwise_and(gray, mask)
    blurred = cv2.GaussianBlur(masked_gray, (15, 15), 0)

    # 2. Find Brightest Point
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
    if max_val < 40:
        return output

    # 3. Localize Tumor
    thresh_val = max_val * 0.80
    _, bright_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN,  kernel)
    cleaned = cv2.morphologyEx(cleaned,     cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        target_contour = None
        for c in contours:
            if cv2.pointPolygonTest(c, (float(max_loc[0]), float(max_loc[1])), False) >= 0:
                target_contour = c
                break
        if target_contour is None:
            target_contour = max(contours, key=cv2.contourArea)

        x, y, bw, bh = cv2.boundingRect(target_contour)
        if (bw * bh) < (h * w * 0.002):
            return output

        # Padding & Drawing
        pad = 10
        x, y = max(0, x - pad), max(0, y - pad)
        bw, bh = min(w - x, bw + pad * 2), min(h - y, bh + pad * 2)

        box_color = {
            'glioma': (255, 80, 80), 'meningioma': (80, 255, 80), 'pituitary': (80, 150, 255)
        }.get(class_name, (255, 255, 0))

        cv2.rectangle(output, (x, y), (x + bw, y + bh), box_color, 2)

        label = f'{class_name} {confidence:.1f}%'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = y - 10 if y - 10 > th else y + bh + th + 10
        cv2.rectangle(output, (x, ly - th - 5), (x + tw + 10, ly + 5), box_color, -1)
        cv2.putText(output, label, (x + 5, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return output

def img_to_base64(img_array):
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ── Chatbot Interaction ───────────────────────────────────────────────
def get_groq_reply(message, history):
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type":  "application/json"
    }

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    messages += history[-10:]   # Last 10 messages for context
    messages.append({"role": "user", "content": message})

    payload = {
        "model":       config.GROQ_MODEL,
        "messages":    messages,
        "max_tokens":  512,
        "temperature": 0.4,
    }

    resp = requests.post(config.GROQ_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']
