# AI Agronomist: Plant Disease Detection

## Problem Description

Plant diseases are a major threat to global food security, causing significant crop losses each year. Early and accurate detection of plant diseases is crucial for farmers to take timely action and minimize damage. Traditional disease identification relies on expert knowledge and visual inspection, which can be time-consuming, subjective, and inaccessible to many farmers.

This project leverages **deep learning and computer vision** to automatically detect and classify plant diseases from leaf images. Using the PlantVillage dataset, we trained convolutional neural network models (Xception and MobileNetV2) to identify **38 different plant disease classes** across 14 crop species.

### Key Features

- **Multi-class Classification**: Identifies 38 different plant conditions (diseases + healthy states)
- **Transfer Learning**: Uses pre-trained models (Xception, MobileNetV2) fine-tuned on plant disease data
- **High Accuracy**: Achieves high accuracy on test data
- **REST API**: Flask-based web service for easy integration
- **Docker Support**: Containerized deployment for production use

### Supported Plants and Diseases

The model can identify diseases in:
- Apple (4 conditions)
- Blueberry (healthy)
- Cherry (2 conditions)
- Corn/Maize (4 conditions)
- Grape (4 conditions)
- Orange (1 disease)
- Peach (2 conditions)
- Pepper (2 conditions)
- Potato (3 conditions)
- Raspberry (healthy)
- Soybean (healthy)
- Squash (1 disease)
- Strawberry (2 conditions)
- Tomato (10 conditions)

## Dataset

This project uses the **PlantVillage Dataset**, which contains 54,305 images of healthy and diseased plant leaves.

### Download Instructions

1. **Option 1: Kaggle**
   ```bash
   # Install kaggle CLI
   pip install kaggle
   
   # Download dataset
   kaggle datasets download -d emmarex/plantdisease
   
   # Unzip to data folder
   unzip plantdisease.zip -d data/
   ```

2. **Option 2: Direct Download**
   - Visit: https://www.kaggle.com/datasets/emmarex/plantdisease
   - Download and extract to `data/plantvillage dataset/color/`

### Dataset Structure
```
data/
└── plantvillage dataset/
    └── color/
        ├── Apple___Apple_scab/
        ├── Apple___Black_rot/
        ├── Apple___Cedar_apple_rust/
        ├── Apple___healthy/
        ├── ... (38 class folders)
        └── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
```

## Project Structure

```
plant-disease-detection/
├── data/                          # Dataset folder (not in repo)
├── images/                        # Project images
│   └── logo.png
├── models/                        # Saved models (created after training)
│   ├── plant_disease_model.keras  # Trained Keras model
│   └── class_indices.json         # Class name mappings
├── notebook.ipynb                 # EDA, model selection, training experiments
├── train.py                       # Final model training script
├── predict.py                     # Flask prediction service
├── requirements.txt               # Python dependencies (pip)
├── pyproject.toml                 # Project dependencies (UV)
├── uv.lock                        # UV lock file
├── Dockerfile                     # Docker container configuration
├── .dockerignore                  # Files to exclude from Docker build
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Installation & Setup

### Prerequisites

- Python 3.11+ (recommended: 3.13)
- CUDA-capable GPU (optional, for faster training)


**Create virtual environment & Activate**
```bash
# Create virtual environment

uv venv

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

**Install dependencies**

```bash
uv sync
```

## Usage

### 1. Training the Model

To train the model from scratch:

```bash
python train.py
```

This will:
- Load and preprocess the PlantVillage dataset
- Train an Xception model with transfer learning
- Save the best model to `models/plant_disease_model.keras`
- Save class indices to `models/class_indices.json`

Training parameters can be configured in `train.py`:
- `EPOCHS_PHASE1`: Feature extraction epochs (default: 10)
- `EPOCHS_PHASE2`: Fine-tuning epochs (default: 10)
- `BATCH_SIZE`: Batch size (default: 32)
- `IMAGE_SIZE`: Input image size (default: 224)

### 2. Running the Prediction Service

Start the Flask API server:

```bash
python predict.py
```

The service will be available at `http://localhost:5000`

### 3. Making Predictions

#### Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Predict Disease
```bash
curl -X POST -F "image=@path/to/leaf_image.jpg" http://localhost:5000/predict
```

Response:
```json
{
  "success": true,
  "prediction": {
    "class": "Tomato___Late_blight",
    "plant": "Tomato",
    "disease": "Late_blight",
    "confidence": 0.9823
  },
  "top_5_predictions": [
    {"class": "Tomato___Late_blight", "confidence": 0.9823},
    {"class": "Tomato___Early_blight", "confidence": 0.0102},
    ...
  ]
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Predict disease from image |
| `/classes` | GET | Get list of all disease classes |


## Docker Deployment

### Build Docker Image

```bash
docker build -t plant-disease-detection .
```

### Run Container

```bash
docker run -p 5000:5000 plant-disease-detection
```

### Test

```bash
curl http://localhost:5000/health
```