<div align="center">

# 🏥 HealthVision AI

### AI-Powered Multi-Disease Health Detection System

<a href="https://healthvision-ai-1enh.onrender.com/" target="_blank"><img src="https://img.shields.io/badge/🌐_Live_Demo-HealthVision_AI-0891b2?style=for-the-badge" alt="Live Demo"></a>
<a href="https://www.python.org/" target="_blank"><img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
<a href="https://www.tensorflow.org/" target="_blank"><img src="https://img.shields.io/badge/TensorFlow-2.16.2-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
<a href="https://flask.palletsprojects.com/" target="_blank"><img src="https://img.shields.io/badge/Flask-Backend-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask"></a>
<a href="https://opencv.org/" target="_blank"><img src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV"></a>
<a href="https://render.com/" target="_blank"><img src="https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=flat-square&logo=render&logoColor=white" alt="Render"></a>

<br>

> A comprehensive AI-based health screening web application that uses deep learning and computer vision to detect diseases from visual analysis of **eyes**, **face/skin**, and **nails**.

**Department of Computer Science & Engineering (Artificial Intelligence)**
**Govt. Engineering College, Munger**

<br>

<a href="https://healthvision-ai-1enh.onrender.com/" target="_blank">🌐 Live Demo</a> · <a href="PROJECT_REPORT.md">📄 Project Report</a> · <a href="https://healthvision-ai-1enh.onrender.com/documentation" target="_blank">📖 Documentation</a>

</div>

---

## 🌐 Live Demo

**🔗 <a href="https://healthvision-ai-1enh.onrender.com/" target="_blank">https://healthvision-ai-1enh.onrender.com/</a>**

> **⚠️ Important — Render Free Tier Cold Start:**
> This project is deployed on **Render's free tier**. Free-tier services **spin down after 15 minutes of inactivity** (sleep mode). When you visit the link after a period of inactivity:
>
> - The **first request may take 1–3 minutes** while the server wakes up and loads the TensorFlow models into memory.
> - Subsequent requests will respond normally and quickly.
> - If the page shows a loading spinner or a timeout error, **please wait and refresh** after a minute.
> - The app runs in **demo mode** on Render (model files are large and may be skipped on the free tier). For full AI inference with real model predictions, run the project locally.
>
> **Tip:** If you plan to demo the app, visit the link a few minutes beforehand so it's warmed up and ready.

---

## 📋 Table of Contents

- [Live Demo](#-live-demo)
- [Abstract](#-abstract)
- [Features](#-features)
- [Detection Modules](#-detection-modules)
- [Technologies Used](#-technologies-used)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [API Endpoints](#-api-endpoints)
- [Model Training](#-model-training)
- [Future Scope](#-future-scope)
- [Team](#-team)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## 📝 Abstract

**HealthVision AI** is an intelligent health screening web application that leverages deep learning and computer vision techniques to assist in the early detection of diseases through visual analysis. The system provides three detection modules — **Eye/Jaundice Detection**, **Face/Skin Disease Detection**, and **Nail Disease Detection** — each powered by transfer learning models trained on medical image datasets.

The platform offers a user-friendly web interface that supports both image upload and live camera capture, processes images using advanced preprocessing techniques (bilateral filtering, CLAHE contrast normalization, Haar cascade detection), and delivers instant predictions with confidence scores, severity levels, and detailed medical information including causes, symptoms, and doctor recommendations.

**Keywords:** Deep Learning, CNN, Transfer Learning, Medical Image Classification, Flask, TensorFlow, OpenCV, Healthcare AI

---

## ✨ Features

| Feature                       | Description                                                                   |
| ----------------------------- | ----------------------------------------------------------------------------- |
| **Three Detection Modules**   | Eye (Jaundice), Face (Skin Diseases), and Nail (Nail Diseases)                |
| **Deep Learning Models**      | Transfer learning with EfficientNetB0 and MobileNetV2 architectures           |
| **Multiple Input Methods**    | File upload (PNG, JPG, JPEG, JFIF, BMP, WebP) and live camera capture         |
| **Advanced Image Processing** | Haar cascade eye detection, bilateral filtering, CLAHE normalization          |
| **Comprehensive Results**     | Disease prediction with confidence scores, top-3 predictions, severity levels |
| **Detailed Health Reports**   | Disease descriptions, causes, symptoms, and doctor recommendations            |
| **Printable Reports**         | Generate and print health assessment reports directly from the browser        |
| **Demo Mode**                 | Fully functional UI demonstration when models are not loaded                  |
| **Responsive Design**         | Works across desktop and mobile devices                                       |
| **Production Ready**          | Configured for deployment on Render with Gunicorn WSGI server                 |

---

## 🔬 Detection Modules

### Module 1: Eye / Jaundice Detection

| Aspect                 | Details                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------- |
| **Model File**         | `jaundice_model.h5`                                                                   |
| **Architecture**       | MobileNet-based CNN (Transfer Learning)                                               |
| **Output**             | Binary — **Jaundice** / **Normal**                                                    |
| **Preprocessing**      | Bilateral filtering, CLAHE, Haar cascade eye detection, HSV/LAB colorspace analysis   |
| **Input Size**         | 224 × 224 pixels                                                                      |
| **Detection Strategy** | Face → Eye ROI (primary) → Direct eye detection (fallback) → Full image (last resort) |

Analyzes the sclera (white part of the eye) for yellow discoloration to detect jaundice. Uses a multi-strategy Haar Cascade pipeline for robust eye localization.

### Module 2: Face / Skin Disease Detection

| Aspect            | Details                                                                    |
| ----------------- | -------------------------------------------------------------------------- |
| **Model File**    | `face_model.h5`                                                            |
| **Architecture**  | EfficientNetB0 (Transfer Learning)                                         |
| **Classes**       | **Acne**, **Eczema**, **Herpes**, **Panu** (Tinea Versicolor), **Rosacea** |
| **Preprocessing** | Bilateral filtering, CLAHE, EfficientNet preprocessing                     |
| **Input Size**    | 224 × 224 pixels                                                           |

Classifies facial skin diseases from photographs with top-3 predictions and confidence scores.

### Module 3: Nail Disease Detection

| Aspect            | Details                                                          |
| ----------------- | ---------------------------------------------------------------- |
| **Model File**    | `nail_model.h5`                                                  |
| **Architecture**  | MobileNetV2 (Transfer Learning)                                  |
| **Classes**       | **Healthy**, **Onychomycosis** (Fungal Infection), **Psoriasis** |
| **Preprocessing** | Bilateral filtering, CLAHE, custom normalization                 |
| **Input Size**    | 224 × 224 pixels                                                 |

Identifies nail health conditions from nail photographs with detailed disease info and recommendations.

### Disease Information Database

The system includes a comprehensive built-in database of **25+ diseases/conditions** with:

- Severity Level (Healthy / Mild / Moderate / High)
- Medical description of the condition
- Known causes and risk factors
- Common signs and symptoms
- Specialist consultation recommendations

---

## 🛠️ Technologies Used

| Category                        | Technology                     |
| ------------------------------- | ------------------------------ |
| **Backend**                     | Python 3.11, Flask, Flask-CORS |
| **Deep Learning**               | TensorFlow 2.16.2, Keras       |
| **Computer Vision**             | OpenCV (Haar Cascades), Pillow |
| **Data Processing**             | NumPy, Matplotlib, Seaborn     |
| **Frontend**                    | HTML5, CSS3, JavaScript (ES6)  |
| **Camera Integration**          | WebRTC, Canvas API             |
| **Client-Server Communication** | Fetch API (AJAX)               |
| **WSGI Server**                 | Gunicorn                       |
| **Deployment**                  | Render (Cloud Platform)        |
| **Model Format**                | HDF5 (.h5)                     |
| **Version Control**             | Git, GitHub                    |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Frontend (Browser)                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Image Upload │  │ Camera Input │  │  UI/Results │ │
│  └──────┬──────┘  └──────┬───────┘  └─────▲──────┘ │
│         │                │                 │         │
│         └────────┬───────┘                 │         │
│                  │ (Fetch API / AJAX)       │         │
└──────────────────┼─────────────────────────┼─────────┘
                   │                         │
                   ▼                         │
┌──────────────────────────────────────────────────────┐
│                Flask Backend (app.py)                 │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │           Image Preprocessing                 │    │
│  │  • Bilateral Filtering (Denoising)            │    │
│  │  • CLAHE Contrast Normalization               │    │
│  │  • Haar Cascade Eye/Face Detection            │    │
│  │  • HSV/LAB Colorspace Analysis                │    │
│  │  • Resize to 224×224                          │    │
│  └──────────────────┬───────────────────────────┘    │
│                     │                                 │
│  ┌──────────────────▼───────────────────────────┐    │
│  │           TensorFlow Models                   │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │    │
│  │  │ Jaundice │ │   Face   │ │    Nail      │  │    │
│  │  │  Model   │ │  Model   │ │   Model      │  │    │
│  │  │  (.h5)   │ │  (.h5)   │ │   (.h5)      │  │    │
│  │  └──────────┘ └──────────┘ └──────────────┘  │    │
│  └──────────────────┬───────────────────────────┘    │
│                     │                                 │
│  ┌──────────────────▼───────────────────────────┐    │
│  │        Disease Information Database           │    │
│  │  • Severity Levels   • Descriptions           │    │
│  │  • Causes            • Symptoms               │    │
│  │  • Recommendations                            │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

### Data Flow

```
User Image → Upload / Camera Capture
           → Server receives image bytes
           → Image Decoding (cv2.imdecode)
           → Image Enhancement (Bilateral Filter + CLAHE)
           → Module-specific preprocessing
           → Model Inference (TensorFlow)
           → Result Aggregation
           → JSON Response (prediction, confidence, disease info)
           → Frontend renders results with severity indicators
```

---

## 📁 Project Structure

```
HealthVision-AI/
│
├── app.py                          # Main Flask application (backend + API)
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render cloud deployment configuration
├── README.md                       # Project documentation (this file)
├── PROJECT_REPORT.md               # Detailed academic project report
│
├── face_model.h5                   # Pre-trained face/skin disease model
├── jaundice_model.h5               # Pre-trained jaundice detection model
├── nail_model.h5                   # Pre-trained nail disease model
│
├── templates/                      # HTML templates
│   ├── index.html                  # Landing page (features, about, team)
│   ├── detect.html                 # Detection interface (upload + camera)
│   ├── documentation.html          # Documentation page
│   ├── documentation_detailed.html # Detailed medical/API documentation
│   └── project_report.html         # Project report page
│
├── Face Dection/                   # Face model training notebook
│   └── Face_dataset.ipynb
│
├── Jaundice Dection/               # Jaundice model training notebook
│   └── Jaundice.ipynb
│
├── Nail Dection/                   # Nail model training notebook
│   └── nail_disease_code.ipynb
│
├── Team/                           # Team member profile photos
│   ├── 01.png                      # Gaurav Kumar
│   ├── 02.png                      # Nitesh Kumar
│   ├── 03.png                      # Rupesh Kumar
│   ├── 04.png                      # Indrajeet Kumar
│   └── 05.png                      # Dr. Saurabh Suman (Supervisor)
│
└── Test Data set/                  # Test datasets
    ├── ezyZip Data Set/            # Jaundice test images (Jaundice/Normal)
    ├── FaceZip Data Set/           # Face test images (acne/eksim/herpes/panu/rosacea)
    └── nailZip Data Set/           # Nail test images (healthy/onychomycosis/psoriasis)
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11.9 or compatible version
- pip (Python package manager)
- Git

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/gauravssah/HealthVision-AI.git
   cd HealthVision-AI
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   # or
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   python app.py
   ```

5. **Open in browser**

   Navigate to `http://localhost:5000`

### Dependencies

```
tensorflow==2.16.2
flask
flask-cors
Pillow
numpy
opencv-python-headless
gunicorn
```

---

## 📖 Usage Guide

### 1. Eye / Jaundice Detection

1. Navigate to the **Detection** page
2. Select the **Eye/Jaundice** tab
3. Upload an eye image or capture via camera
4. Click **Analyze** to get the prediction
5. View results — jaundice probability, detected eye crop, severity, and health report

### 2. Face / Skin Disease Detection

1. Select the **Face/Skin** tab
2. Upload a face image or use camera capture
3. The system analyzes for: **Acne**, **Eczema**, **Herpes**, **Panu** (Tinea Versicolor), and **Rosacea**
4. View top-3 predictions with confidence scores and detailed disease info

### 3. Nail Disease Detection

1. Select the **Nail** tab
2. Upload a nail image or use camera capture
3. The system analyzes for: **Healthy**, **Onychomycosis**, **Psoriasis**, and additional conditions
4. View detailed disease information, severity, and specialist recommendations

### Supported Image Formats

`PNG` · `JPG` · `JPEG` · `JFIF` · `BMP` · `WebP`

---

## 🔌 API Endpoints

| Route                     | Method | Description                                                                     |
| ------------------------- | ------ | ------------------------------------------------------------------------------- |
| `/`                       | GET    | Landing page                                                                    |
| `/detect`                 | GET    | Detection interface                                                             |
| `/documentation`          | GET    | Documentation page                                                              |
| `/documentation/detailed` | GET    | Detailed medical & API documentation                                            |
| `/team/<filename>`        | GET    | Team member profile images                                                      |
| `/predict/eye`            | POST   | Eye/Jaundice detection — returns prediction, confidence, eye crop, disease info |
| `/predict/face`           | POST   | Face/Skin disease detection — returns top-3 predictions with disease info       |
| `/predict/nail`           | POST   | Nail disease detection — returns prediction with disease info                   |

<details>
<summary><strong>Example API Response (Eye Detection)</strong></summary>

```json
{
  "prediction": "Normal",
  "confidence": 95.2,
  "eye_crop": "<base64_encoded_image>",
  "disease_info": {
    "severity": "healthy",
    "description": "No signs of jaundice detected.",
    "causes": "N/A",
    "symptoms": "No abnormal symptoms detected.",
    "recommendation": "Continue regular health checkups."
  }
}
```

</details>

---

## 🧪 Model Training

The training notebooks are available in the repository:

| Module                 | Notebook                  | Directory           |
| ---------------------- | ------------------------- | ------------------- |
| Face/Skin Detection    | `Face_dataset.ipynb`      | `Face Dection/`     |
| Jaundice Detection     | `Jaundice.ipynb`          | `Jaundice Dection/` |
| Nail Disease Detection | `nail_disease_code.ipynb` | `Nail Dection/`     |

### Training Approach

| Aspect                | Details                                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| **Strategy**          | Transfer Learning — pre-trained models (EfficientNetB0, MobileNetV2) fine-tuned on medical image datasets |
| **Data Augmentation** | Rotation, flipping, zoom, brightness adjustment to improve generalization                                 |
| **Optimizer**         | Adam optimizer with learning rate scheduling                                                              |
| **Loss Functions**    | Categorical cross-entropy (multi-class) · Binary cross-entropy (binary classification)                    |
| **Input Size**        | 224 × 224 × 3 (RGB)                                                                                       |

---

## 🔮 Future Scope

- **Additional Disease Modules** — Expand to other body parts and disease categories
- **Model Improvement** — Train on larger, more diverse medical datasets for higher accuracy
- **Mobile Application** — Develop native Android/iOS applications
- **Multi-Language Support** — Add support for regional languages
- **Patient History Tracking** — Allow users to maintain a health screening history
- **Doctor Integration** — Connect with healthcare professionals for follow-up consultations
- **Cloud-Based Model Serving** — Use TensorFlow Serving for scalable inference
- **Real-Time Video Analysis** — Continuous detection from live video feed

---

## 👥 Team

<div align="center">

> **Department of Computer Science & Engineering (Artificial Intelligence)**
> **Govt. Engineering College, Munger**
> **Final Year Minor Project (2025–2026)**

</div>

### 🎓 Project Supervisor

<div align="center">
<table>
  <tr>
    <td align="center" width="280">
      <a href="https://www.linkedin.com/in/dr-saurabh-suman-409a4697/" target="_blank">
        <img src="Team/05.png" width="150" height="150" alt="Dr. Saurabh Suman">
      </a>
      <br><br>
      <strong>Dr. Saurabh Suman</strong>
      <br>
      <sub>🏫 Assistant Professor</sub>
      <br>
      <sub>Dept. of CSE, Govt. Engineering College, Munger</sub>
      <br><br>
      <a href="https://www.linkedin.com/in/dr-saurabh-suman-409a4697/" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
      </a>
    </td>
  </tr>
</table>
</div>

### 👨‍💻 Student Developers

<div align="center">
<table>
  <tr>
    <td align="center" width="220">
      <a href="https://www.linkedin.com/in/gauravssah" target="_blank">
        <img src="Team/01.png" width="120" height="120" alt="Gaurav Kumar">
      </a>
      <br><br>
      <strong>Gaurav Kumar</strong>
      <br>
      <sub>🤖 CSE (Artificial Intelligence)</sub>
      <br>
      <sub>📅 Session: 2023–2026</sub>
      <br><br>
      <a href="https://www.linkedin.com/in/gauravssah" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
      </a>
    </td>
    <td align="center" width="220">
      <a href="https://www.linkedin.com/in/nitesh-kumar-pandey-211136260/" target="_blank">
        <img src="Team/02.png" width="120" height="120" alt="Nitesh Kumar">
      </a>
      <br><br>
      <strong>Nitesh Kumar</strong>
      <br>
      <sub>🤖 CSE (Artificial Intelligence)</sub>
      <br>
      <sub>📅 Session: 2022–2026</sub>
      <br><br>
      <a href="https://www.linkedin.com/in/nitesh-kumar-pandey-211136260/" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
      </a>
    </td>
    <td align="center" width="220">
      <a href="https://www.linkedin.com/in/rupeskumar" target="_blank">
        <img src="Team/03.png" width="120" height="120" alt="Rupesh Kumar">
      </a>
      <br><br>
      <strong>Rupesh Kumar</strong>
      <br>
      <sub>🤖 CSE (Artificial Intelligence)</sub>
      <br>
      <sub>📅 Session: 2022–2026</sub>
      <br><br>
      <a href="https://www.linkedin.com/in/rupeskumar" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
      </a>
    </td>
    <td align="center" width="220">
      <a href="https://www.linkedin.com/in/indrajeetkumar01/" target="_blank">
        <img src="Team/04.png" width="120" height="120" alt="Indrajeet Kumar">
      </a>
      <br><br>
      <strong>Indrajeet Kumar</strong>
      <br>
      <sub>🤖 CSE (Artificial Intelligence)</sub>
      <br>
      <sub>📅 Session: 2023–2026</sub>
      <br><br>
      <a href="https://www.linkedin.com/in/indrajeetkumar01/" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
      </a>
    </td>
  </tr>
</table>
</div>

---

## ⚠️ Disclaimer

> **Medical Disclaimer:** HealthVision AI is an **academic research project** and preliminary screening tool only. It is **NOT** a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider for any medical concerns. The predictions provided by this system are indicative and should be verified by clinical examination.

---

## 📄 License

This project is developed as a final-year **Minor Project** at **Govt. Engineering College, Munger** under the **Department of Computer Science & Engineering (Artificial Intelligence)**.

---

<div align="center">

**🏥 HealthVision AI — Empowering Health Through Artificial Intelligence**

<a href="https://healthvision-ai-1enh.onrender.com/" target="_blank"><img src="https://img.shields.io/badge/🌐_Try_it_Live-HealthVision_AI-0891b2?style=for-the-badge" alt="Live Demo"></a>

</div>
