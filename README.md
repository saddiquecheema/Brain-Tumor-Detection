# 🧠 Brain Tumor MRI Classification System

An end-to-end AI-powered web application that classifies brain MRI scans into **4 categories** using Deep Learning and Computer Vision.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)
![Keras](https://img.shields.io/badge/Keras-3.x-red?logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)

---

## 📌 Overview

This system uses **EfficientNetB0** (Transfer Learning) to detect brain tumors from MRI images. A user uploads an MRI scan through a web interface, and the model instantly classifies it with confidence scores. The tumor region is automatically highlighted using **OpenCV**, and an integrated **AI chatbot (NeuroGuide)** powered by **Groq (Llama 3.3 70B)** answers medical follow-up questions.

---

## 🎯 Tumor Classes

| Class | Description |
|---|---|
| 🔴 **Glioma** | Tumor inside brain tissue — most serious |
| 🟠 **Meningioma** | Tumor on brain covering — often operable |
| 🟡 **Pituitary** | Tumor in pituitary gland — affects hormones |
| 🟢 **No Tumor** | Normal healthy MRI scan |

---

## ✨ Features

- ✅ Real-time MRI image classification (4 classes)
- ✅ Tumor region highlighted with bounding box (OpenCV)
- ✅ Confidence scores for all classes with probability bars
- ✅ Integrated AI medical chatbot (NeuroGuide — Groq LLM)
- ✅ Original + annotated image side by side
- ✅ Medical description and advice per prediction
- ✅ Fully responsive dark-theme web UI
- ✅ CPU compatible — no GPU required

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Deep Learning** | TensorFlow · Keras · EfficientNetB0 |
| **Computer Vision** | OpenCV |
| **Backend** | Python · Flask |
| **AI Chatbot** | Groq API · Llama 3.3 70B |
| **Frontend** | HTML · CSS · JavaScript |
| **Training** | Google Colab (Tesla T4 GPU) |
| **Dataset** | Kaggle Brain Tumor MRI Dataset (7,023 images) |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Architecture | EfficientNetB0 (Transfer Learning) |
| Dataset | 7,023 MRI Images |
| Training | Google Colab (T4 GPU) |
| Accuracy | 97%+ |
| Input Size | 224 × 224 × 3 |
| Classes | 4 |

---

## 📁 Project Structure

```
Brain_tumor_detection/
│
├── app.py                              ← Main Flask application
├── requirements.txt                    ← Python dependencies
├── class_indices.json                  ← Class name mappings
├── .env                                ← API keys (create manually)
│
├── models/
│   ├── efficientnetb0_Fine-Tune.keras
│   └── efficientnetb0_Fine-Tune.weights.h5
│
└── templates/
    └── index.html                      ← Web UI
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Groq API Key
Create `.env` file in root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get free key → https://console.groq.com

### 5. Run
```bash
python app.py
```

### 6. Open browser
```
http://127.0.0.1:5000
```

---

## 🖥️ How to Use

1. Open `http://127.0.0.1:5000`
2. Upload a brain MRI image (JPG / PNG / BMP)
3. Click **Analyze MRI**
4. View predicted class, confidence scores, annotated image
5. Ask follow-up questions to **NeuroGuide AI chatbot**

---

## 🏋️ Model Training (Google Colab)

**Phase 1 — Frozen base (15 epochs)**
- EfficientNetB0 base frozen
- Only classification head trained
- LR: `1e-3`

**Phase 2 — Fine-tuning (10 epochs)**
- Top 50 layers unfrozen
- LR: `1e-5`

**Dataset:** [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## ⚠️ Disclaimer

> This system is for **educational and research purposes only**.
> Always consult a qualified **Neurologist or Radiologist** for medical decisions.

---

## 👨‍💻 Author

**Muhammad Saddique**
- LinkedIn: [muhammad-saddique](https://www.linkedin.com/in/muhammad-saddique-0b4967275)
- Email: Sadddique786@gmail.com