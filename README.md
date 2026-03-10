# HealthVision AI - AI-Powered Multi-Disease Health Detection System

<div align="center">

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![College](https://img.shields.io/badge/College-GECM-blue)
![Academic](https://img.shields.io/badge/Type-Academic%20Project-purple)

**AI-powered web application for detecting jaundice, skin diseases, and nail disorders from medical images**

### 🏫 Government Engineering College, Munger
**Department of Computer Science & Engineering (Artificial Intelligence)**

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Models](#models)
- [Screenshots](#screenshots)
- [Deployment](#deployment)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)
- [License](#license)

---

## 🎯 Overview

**HealthVision AI** is a comprehensive healthcare AI system that leverages Deep Learning and Computer Vision to provide preliminary health assessments through medical image analysis. The application integrates three specialized detection modules to identify health conditions that are visible through images.

### Key Use Cases:
- **Early Detection**: Preliminary health screening from home
- **Healthcare Access**: Bridge the gap for rural/remote areas with limited medical resources
- **Patient Education**: Provide detailed information about detected conditions
- **Medical Consultation**: Guide users to seek professional medical advice

---

## ✨ Features

### 🔍 Three Detection Modules

#### 1. **Jaundice Detection**
- Analyzes eye/sclera images for yellow discoloration
- Indicates potential liver dysfunction or hemolytic disorders
- Provides bilirubin level severity assessment

#### 2. **Skin/Face Disease Detection**
- Classifies 5 common skin conditions:
  - **Acne** - Clogged pores and inflammation
  - **Eczema** - Chronic inflammatory skin condition
  - **Herpes** - Viral infection causing blisters
  - **Panu** - Fungal infection (Tinea Versicolor)
  - **Rosacea** - Facial redness with visible blood vessels

#### 3. **Nail Disease Detection**
- Identifies 3 nail health conditions:
  - **Healthy** - Normal nail appearance
  - **Onychomycosis** - Fungal nail infection
  - **Psoriasis** - Autoimmune nail disorder

### 📱 User Interface
- **Image Upload**: Upload images from device storage
- **Live Camera Capture**: Real-time image capture from webcam
- **Detailed Results**: Disease name, confidence score, severity level
- **Medical Information**: Description, causes, symptoms, and recommendations
- **Responsive Design**: Mobile-friendly interface

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Web Interface (HTML/CSS/JS)           │
│                  (detection.html, index.html)           │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    Flask Backend (app.py)               │
│  - Request handling                                     │
│  - Image preprocessing                                 │
│  - Model inference coordination                        │
│  - CORS handling                                        │
└────────┬──────────────────┬──────────────────┬──────────┘
         │                  │                  │
    ┌────▼──────┐      ┌────▼──────┐      ┌───▼──────┐
    │ TensorFlow│      │ TensorFlow │      │TensorFlow│
    │  Models   │      │  Models    │      │  Models  │
    └────┬──────┘      └────┬──────┘      └───┬──────┘
         │                  │                  │
    ┌────▼──────┐      ┌────▼──────┐      ┌───▼──────┐
    │ Jaundice  │      │ Face/Skin  │      │   Nail   │
    │ Detection │      │ Diseases   │      │ Diseases │
    │(model.h5) │      │(model.h5)  │      │(model.h5)│
    └───────────┘      └────────────┘      └──────────┘
         │ (CNN)             │ (CNN)             │ (CNN)
         └─────────────────────────────────────┘
                      │
           ┌──────────▼──────────┐
           │ Image Preprocessing │
           │ - Resize (224x224)  │
           │ - Normalize         │
           │ - Color Conversion  │
           └─────────────────────┘
                      │
           ┌──────────▼──────────┐
           │   OpenCV (Image     │
           │   Processing)       │
           └─────────────────────┘
```

---

## 🛠️ Technology Stack

### Backend
| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Flask | Latest |
| Deep Learning | TensorFlow/Keras | 2.20.0 |
| Computer Vision | OpenCV | Latest |
| Image Processing | Pillow | Latest |
| CORS | Flask-CORS | Latest |
| Server | Gunicorn | Latest |
| Numerical Computing | NumPy | Latest |

### Frontend
- **HTML5** - Page structure
- **CSS3** - Responsive styling
- **JavaScript (Vanilla)** - Interactive features
- **Bootstrap** - UI components

### Deployment
- **Cloud Platform**: Render
- **Runtime**: Python 3.8+
- **Configuration**: render.yaml

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git
- Virtual Environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/healthvision-ai.git
cd healthvision-ai
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python app.py
```

The application should start on `http://localhost:5000`

---

## 🚀 Usage

### Starting the Application

#### Local Development
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

# Run the application
python app.py
```

### Using the Web Interface

1. **Open Browser**: Navigate to `http://localhost:5000`

2. **Select Detection Type**:
   - Click on "Jaundice Detection"
   - Click on "Skin Disease Detection"
   - Click on "Nail Disease Detection"

3. **Upload Image or Use Camera**:
   - **Option A**: Click "Upload Image" and select from device
   - **Option B**: Click "Use Camera" for real-time capture

4. **View Results**:
   - Disease classification with confidence score
   - Severity level assessment
   - Detailed medical information
   - Recommendations for action

### API Endpoints

#### Jaundice Detection
```http
POST /api/detect_jaundice
Content-Type: multipart/form-data

Parameters:
- image: Image file (png, jpg, jpeg, jfif, bmp, webp)

Response:
{
  "disease": "Jaundice",
  "confidence": 0.95,
  "severity": "moderate",
  "description": "...",
  "causes": "...",
  "symptoms": "...",
  "recommendation": "..."
}
```

#### Skin Disease Detection
```http
POST /api/detect_skin_disease
Content-Type: multipart/form-data

Parameters:
- image: Image file

Response:
{
  "disease": "Acne",
  "confidence": 0.92,
  "severity": "mild",
  "description": "...",
  "causes": "...",
  "symptoms": "...",
  "recommendation": "..."
}
```

#### Nail Disease Detection
```http
POST /api/detect_nail_disease
Content-Type: multipart/form-data

Parameters:
- image: Image file

Response:
{
  "disease": "Onychomycosis",
  "confidence": 0.88,
  "severity": "moderate",
  "description": "...",
  "causes": "...",
  "symptoms": "...",
  "recommendation": "..."
}
```

---

## 📁 Project Structure

```
healthvision-ai/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render deployment config
├── .gitignore                      # Git ignore rules
├── README.md                       # Project documentation
├── PROJECT_REPORT.md               # Detailed project report
│
├── face_model.h5                   # Pre-trained face disease model
├── jaundice_model.h5               # Pre-trained jaundice model
├── nail_model.h5                   # Pre-trained nail disease model
│
├── templates/                      # HTML templates
│   ├── index.html                  # Home page
│   ├── detect.html                 # Detection interface
│   ├── documentation.html          # Quick documentation
│   ├── documentation_detailed.html # Detailed documentation
│   └── project_report.html         # Project report page
│
├── Face Detection/                 # Face disease detection module
│   └── Face_dataset.ipynb          # Data processing notebook
│
├── Jaundice Detection/             # Jaundice detection module
│   └── Jaundice.ipynb              # Model development notebook
│
├── Nail Detection/                 # Nail disease detection module
│   └── nail_disease_code.ipynb     # Model development notebook
│
├── Test Data set/                  # Test datasets
│   ├── ezyZip Data Set/
│   │   ├── Jaundice/
│   │   └── Normal/
│   ├── FaceZip Data Set/
│   │   └── train/
│   │       ├── acne/
│   │       ├── eksim/
│   │       ├── herpes/
│   │       ├── panu/
│   │       └── rosacea/
│   └── nailZip Data Set/
│       ├── healthy/
│       ├── onychomycosis/
│       └── psoriasis/
│
├── Test Images forEye/             # Eye test images
│   └── images.jfif
│
└── Team/                           # Team documentation
```

---

## 📊 Datasets

### Dataset Overview

| Disease Type | Classes | Total Images | Training | Testing |
|-------------|---------|------------|----------|---------|
| **Face/Skin** | 5 | ~2000+ | 70% | 30% |
| **Jaundice** | 2 | ~500+ | 70% | 30% |
| **Nail** | 3 | ~600+ | 70% | 30% |

### Skin Disease Classes
1. **Acne** - Inflammatory condition with pimples and blackheads
2. **Eczema** - Chronic dry and itchy skin
3. **Herpes** - Viral infection with blisters
4. **Panu** - Fungal infection with hypopigmentation
5. **Rosacea** - Facial redness with visible blood vessels

### Jaundice Classes
1. **Jaundiced** - Yellow discoloration of sclera
2. **Normal** - Healthy eye appearance

### Nail Disease Classes
1. **Healthy** - Normal nail appearance
2. **Onychomycosis** - Fungal nail infection
3. **Psoriasis** - Nail damage due to autoimmune disorder

---

## 🤖 Models

### Model Architecture

All models use **Convolutional Neural Networks (CNNs)** for image classification:

#### Model Specifications
```
Input Layer:
├─ Image Size: 224 × 224 pixels
├─ Color Channels: 3 (RGB)
└─ Normalization: Applied

Hidden Layers:
├─ Convolutional Blocks
├─ ReLU Activation
├─ Max Pooling
├─ Batch Normalization
├─ Dropout (0.5)
└─ Dense Layers

Output Layer:
├─ Softmax Activation
├─ Confidence Scores (0-1)
└─ Class Probability Distribution
```

### Training Details
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Batch Size**: 32
- **Epochs**: 50-100
- **Validation Split**: 20%

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Face Disease | 92.5% | 91.8% | 92.3% | 92.0% |
| Jaundice | 94.2% | 93.9% | 94.5% | 94.2% |
| Nail Disease | 89.7% | 89.1% | 90.2% | 89.6% |

---

## 📸 Screenshots

### Home Page
- Clean, intuitive interface
- Three detection module options
- Quick navigation

### Detection Interface
- Image upload zone
- Camera capture button
- Real-time preview

### Results Page
- Disease classification
- Confidence percentage
- Severity assessment
- Medical information card
- Recommendations section

---

## 🚀 Deployment

### Deploy on Render

1. **Push to GitHub**:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Create Render Account**: Visit https://render.com

3. **Connect GitHub Repository**:
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select the repository

4. **Configure Build Settings**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Python Version**: 3.8+

5. **Environment Variables** (if needed):
   - Set any required environment variables in Render dashboard

6. **Deploy**: Click "Create Web Service"

### Live Demo
Once deployed, access your application at:
```
https://your-app-name.onrender.com
```

---

## 🔄 Model Inference Flow

```
1. User uploads image or captures from camera
                    ↓
2. Image validation and preprocessing
   - Format check
   - Size verification
   - RGB conversion if needed
                    ↓
3. Image resizing to 224×224 pixels
                    ↓
4. Normalization (pixel values 0-1)
                    ↓
5. Load appropriate pre-trained model
   - Based on disease type selected
   - Use custom objects for compatibility
                    ↓
6. Model inference (forward pass)
   - Get prediction probabilities
   - Calculate confidence score
                    ↓
7. Post-processing
   - Get class label
   - Determine severity level
   - Retrieve disease information
                    ↓
8. Format and return JSON response
   - Disease name
   - Confidence percentage
   - Medical details
   - Recommendations
                    ↓
9. Display results in web interface
```

---

## 📚 Detailed Disease Information

### Jaundice
**Description**: Yellow discoloration of skin and whites of eyes due to elevated bilirubin  
**Causes**: Liver disease, bile duct obstruction, hemolytic anemia  
**Symptoms**: Yellow skin/eyes, dark urine, fatigue, abdominal pain  
**Recommendation**: Consult gastroenterologist, blood tests, liver function assessment  

### Acne
**Description**: Common skin condition with clogged pores and inflammation  
**Causes**: Excess sebum, bacterial growth, hormonal changes  
**Symptoms**: Pimples, blackheads, whiteheads, redness  
**Recommendation**: Gentle cleansing, topical treatments, dermatologist consultation  

### Eczema
**Description**: Chronic inflammatory skin disorder causing dryness and itching  
**Causes**: Genetic factors, immune dysfunction, environmental triggers  
**Symptoms**: Red, dry, itchy patches, sometimes with weeping  
**Recommendation**: Regular moisturizing, trigger avoidance, topical corticosteroids  

### Herpes
**Description**: Viral infection causing painful blisters and sores  
**Causes**: Herpes Simplex Virus (HSV-1 or HSV-2)  
**Symptoms**: Painful blisters, tingling, swollen lymph nodes, fever  
**Recommendation**: Antiviral medications, healthcare provider consultation  

### Panu
**Description**: Fungal skin infection with discolored patches  
**Causes**: Malassezia fungus overgrowth in warm, humid conditions  
**Symptoms**: Light or dark patches, mild itching  
**Recommendation**: Antifungal creams, keep skin dry, dermatologist follow-up  

### Rosacea
**Description**: Chronic skin condition with facial redness and visible vessels  
**Causes**: Genetic factors, vascular instability, environmental triggers  
**Symptoms**: Facial flushing, visible blood vessels, small red bumps  
**Recommendation**: Avoid triggers, topical treatments, dermatologist care  

### Onychomycosis
**Description**: Fungal nail infection causing discoloration and thickness  
**Causes**: Fungal infection, poor hygiene, nail trauma  
**Symptoms**: Discolored nails, thickening, crumbling, separation  
**Recommendation**: Antifungal medications, nail care, patience for regrowth  

### Psoriasis
**Description**: Autoimmune disorder affecting nail structure and appearance  
**Causes**: Genetic factors, immune system dysfunction  
**Symptoms**: Nail pitting, discoloration, thickening, separation  
**Recommendation**: Topical treatments, systemic therapy, dermatologist coordination  

---

## 🔐 Security Features

- **Input Validation**: File type and size verification
- **CORS Protection**: Configured CORS headers
- **Error Handling**: Graceful error responses
- **Secure File Handling**: Temporary file cleanup
- **Model Security**: Loaded with custom object handling

---

## ⚙️ Configuration

### Image Settings
```python
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "jfif", "bmp", "webp"}
IMG_SIZE = (224, 224)  # Model input size
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
```

### Model Loading
- All models loaded with TensorFlow 2.20.0
- Custom DepthwiseConv2D layer handling for compatibility
- Lazy loading for performance optimization

---

## 🐛 Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Run `pip install -r requirements.txt` |
| Model file not found | Ensure `.h5` files are in project root |
| Port 5000 already in use | Change port in app.py or kill process |
| CORS errors | Verify Flask-CORS is installed |
| Image upload fails | Check file format and size limits |
| Model slow on startup | First inference is slower, subsequent ones are faster |

---

## 🚀 Future Enhancements

1. **Multi-Image Batch Processing**: Process multiple images simultaneously
2. **Advanced Analytics**: Patient history tracking and trends
3. **Mobile App**: Native iOS/Android applications
4. **Real-time Monitoring**: Continuous health metric tracking
5. **Specialist Integration**: Direct connection to healthcare providers
6. **Multi-language Support**: Localization for global users
7. **Advanced Models**: Integration of more disease types
8. **Confidence Calibration**: Uncertainty estimation
9. **Explainability**: Grad-CAM visualization for model decisions
10. **Blockchain Security**: Medical record encryption and storage

---

## 📈 Performance Metrics

### Response Times
- Model Load Time: ~2-3 seconds (first load)
- Image Processing: ~0.1 seconds
- Model Inference: ~0.5-1 second
- Total Response Time: ~1-2 seconds

### System Requirements
- **RAM**: Minimum 2GB, Recommended 4GB+
- **Storage**: ~1GB for models and dependencies
- **CPU**: Modern processor recommended
- **GPU**: Optional (accelerates inference)

---

## 👥 Contributors & Team

### 📚 Academic Institution
- **College**: Government Engineering College, Munger
- **Department**: Computer Science & Engineering (Artificial Intelligence)
- **Location**: Munger, Bihar, India

### 👨‍🏫 Project Supervisor
| Name | Position | Department | Contact |
|------|----------|-----------|---------|
| **Dr. Saurabh Suman** | Assistant Professor & Project Supervisor | CSE (AI) | [LinkedIn](https://www.linkedin.com/in/dr-saurabh-suman-409a4697/) |

### 👨‍💼 Team Members

| # | Name | Registration No. | Branch | LinkedIn |
|---|------|-----------------|--------|----------|
| 1 | **Gaurav Kumar** | 23151144901 | CSE (AI) | [LinkedIn](https://www.linkedin.com/in/gauravssah) |
| 2 | **Nitesh Kumar** | 22151144040 | CSE (AI) | [LinkedIn](https://www.linkedin.com/in/nitesh-kumar-pandey-211136260/) |
| 3 | **Rupesh Kumar** | 22151144010 | CSE (AI) | [LinkedIn](https://www.linkedin.com/in/rupeskumar) |
| 4 | **Indrajeet Kumar** | 23151144906 | CSE (AI) | [LinkedIn](https://www.linkedin.com/in/indrajeetkumar01/) |

### 🎯 Team Contributions

- **Gaurav Kumar**: Deep Learning Model Development & Optimization
- **Nitesh Kumar**: Backend Development & API Architecture (Flask)
- **Rupesh Kumar**: Frontend Development & UI/UX Design
- **Indrajeet Kumar**: Computer Vision & Image Processing (OpenCV)
- **Dr. Saurabh Suman**: Project Guidance, Medical Accuracy Verification, Research Supervision

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### License Terms
- ✅ Free for commercial use
- ✅ Modification allowed
- ✅ Distribution allowed
- ℹ️ Include license and copyright notice

---

## ⚠️ Disclaimer

**Important**: This application is designed for **preliminary screening and educational purposes only**. 

**NOT a substitute for:**
- Professional medical diagnosis
- Clinical examination
- Healthcare provider consultation
- Medical treatment

**Always consult a qualified healthcare professional for:**
- Accurate diagnosis
- Treatment recommendations
- Medical decisions

---

## 📞 Support & Contact

- **Issues**: Create an issue on GitHub
- **Questions**: Open a discussion on GitHub
- **Enhancement Requests**: Feature request on GitHub
- **Email**: [Your Email]

---

## 🙏 Acknowledgments

- TensorFlow/Keras team for excellent deep learning framework
- Flask developers for robust web framework
- OpenCV community for computer vision tools
- Dataset contributors and medical consultants
- Render for cloud deployment platform

---

## 📝 References

1. Esteva, A., Kuprel, B., Novoa, R. A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115-118.

2. Gulshan, V., Peng, L., Coram, M., et al. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. JAMA, 316(22), 2402-2410.

3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.

5. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

---

<div align="center">

**Made with ❤️ for Healthcare Innovation**

⭐ Star this repository if you find it helpful!

</div>

---

**Last Updated**: March 2026  
**Version**: 1.0.0  
**Status**: Active & Maintained
