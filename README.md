# Brain Tumor Detection System 🧠

This is a web-based application that uses Deep Learning to detect and classify brain tumors from MRI images. The system is built with Flask and TensorFlow, providing real-time predictions and visual annotations of detected tumors.

## 🚀 Features
- **Tumor Classification:** Detects 4 types: Glioma, Meningioma, Pituitary, and No Tumor.
- **Visual Annotation:** Automatically draws bounding boxes around detected tumors using OpenCV.
- **Probability Breakdown:** Shows the confidence level for all possible tumor types.
- **Medical Advice:** Provides a brief description and recommended next steps based on the diagnosis.
- **Interactive UI:** Clean and modern interface for easy image uploads and result viewing.

## 🛠️ Technologies Used
- **Backend:** Flask (Python)
- **Deep Learning:** TensorFlow / Keras (MobileNetV2)
- **Image Processing:** OpenCV, Pillow (PIL), NumPy
- **Frontend:** HTML5, CSS3, JavaScript (Fetch API)

## 📋 Prerequisites
Ensure you have Python 3.9+ installed on your system.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/saddiquecheema/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install flask tensorflow pillow numpy opencv-python
   ```

4. **Ensure Model Files are Present:**
   Make sure `brain_tumor_model.keras` and `class_indices.json` are in the root directory.

## 🏃 Running the App
1. Start the Flask server:
   ```bash
   python app_fixed.py
   ```
2. Open your browser and navigate to:
   `http://localhost:5000`

## 📁 Project Structure
```text
├── app_fixed.py            # Main Flask application
├── brain_tumor_model.keras  # Trained Deep Learning model
├── class_indices.json      # Mapping of indices to tumor classes
├── static/                 # CSS and JS files
├── templates/              # HTML templates
├── .gitignore              # Files to ignore in Git
└── README.md               # Project documentation
```

## ⚠️ Disclaimer
*This application is for educational and demonstration purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.*

---
Developed by [Saddique Cheema](https://github.com/saddiquecheema)
