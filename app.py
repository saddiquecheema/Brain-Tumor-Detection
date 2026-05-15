import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Import custom modules
import config
import logic

# Initialize Flask
app = Flask(__name__)

# Global variables for model and indices
model = None
class_indices = None
index_to_class = None

def init_model():
    global model, class_indices, index_to_class
    if model is None:
        model = logic.load_brain_tumor_model()
        class_indices = logic.load_class_indices()
        index_to_class = {v: k for k, v in class_indices.items()}
        
        if model is None:
            print("[CRITICAL] Model could not be loaded. Check your configurations.")

# In Flask's debug mode with reloader, the code runs twice.
# We only want the HEAVY model loading to happen in the actual worker process.
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
    init_model()

# ── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ping')
def ping():
    return "Server is alive!", 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Safety check: if model isn't loaded yet
    if model is None:
        init_model()
        if model is None:
            return jsonify({'error': 'Model could not be loaded on the server.'}), 500

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Load and preprocess image
        img_bytes = file.read()
        img_pil   = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        orig_img  = np.array(img_pil)

        img_resized = img_pil.resize((224, 224), Image.LANCZOS)
        img_array   = np.expand_dims(np.array(img_resized), axis=0).astype(np.float32)
        
        # Inference
        preds       = model.predict(img_array, verbose=0)
        class_idx   = int(np.argmax(preds[0]))
        class_name  = index_to_class[class_idx]
        confidence  = float(preds[0][class_idx] * 100)

        all_probs = {
            index_to_class[i]: round(float(preds[0][i] * 100), 2)
            for i in range(len(index_to_class))
        }

        # Annotate
        annotated     = logic.annotate_image(orig_img, class_name, confidence)
        annotated_b64 = logic.img_to_base64(annotated)
        original_b64  = logic.img_to_base64(orig_img)

        return jsonify({
            'class':       class_name,
            'confidence':  round(confidence, 2),
            'description': config.DESCRIPTIONS.get(class_name, ''),
            'advice':      config.ADVICE.get(class_name, ''),
            'all_probs':   all_probs,
            'original':    original_b64,
            'annotated':   annotated_b64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message found'}), 400

    user_message = data['message'].strip()
    history      = data.get('history', [])

    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    try:
        reply = logic.get_groq_reply(user_message, history)
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': f'Chatbot error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n[STARTUP] Brain Tumor Detection System")
    print(f"[INFO] Flask server running on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)