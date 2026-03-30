"""
Flask prediction service for Plant Disease Detection.

This script provides a REST API for plant disease classification.

Endpoints:
    GET  /health  - Health check
    GET  /classes - Get list of all disease classes
    POST /predict - Predict disease from leaf image

Usage:
    python predict.py
    
    # Or with auto-reload for development:
    flask --app predict run --reload
"""

import os
import json
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

import tensorflow as tf
from tensorflow.keras.models import load_model

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = os.environ.get('MODEL_PATH', './models/plant_disease_model.keras')
CLASS_INDICES_PATH = os.environ.get('CLASS_INDICES_PATH', './models/class_indices.json')
IMAGE_SIZE = 224
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

# ============================================================
# Initialize Flask app
# ============================================================
app = Flask(__name__)

# Global variables for model and class indices
model = None
class_indices = None


def load_model_and_classes():
    """Load the trained model and class indices."""
    global model, class_indices
    
    print("Loading model...")
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")
        model = None
    
    print("Loading class indices...")
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        # Convert string keys to int
        class_indices = {int(k): v for k, v in class_indices.items()}
        print(f"Loaded {len(class_indices)} classes")
    else:
        print(f"Warning: Class indices not found at {CLASS_INDICES_PATH}")
        class_indices = None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for model prediction.
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed numpy array ready for prediction
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def parse_class_name(class_name: str) -> dict:
    """
    Parse class name into plant and disease components.
    
    Args:
        class_name: Class name in format "Plant___Disease"
        
    Returns:
        Dictionary with plant and disease names
    """
    if "___" in class_name:
        plant, disease = class_name.split("___")
        return {
            "plant": plant.replace("_", " "),
            "disease": disease.replace("_", " ")
        }
    return {
        "plant": class_name,
        "disease": "Unknown"
    }


# ============================================================
# API Endpoints
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_loaded": class_indices is not None
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of all disease classes."""
    if class_indices is None:
        return jsonify({
            "success": False,
            "error": "Class indices not loaded"
        }), 500
    
    classes = []
    for idx, class_name in class_indices.items():
        parsed = parse_class_name(class_name)
        classes.append({
            "index": idx,
            "class": class_name,
            "plant": parsed["plant"],
            "disease": parsed["disease"]
        })
    
    return jsonify({
        "success": True,
        "total_classes": len(classes),
        "classes": sorted(classes, key=lambda x: x["class"])
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict plant disease from uploaded image.
    
    Expects multipart/form-data with 'image' file field.
    
    Returns:
        JSON with prediction results including:
        - Top prediction with confidence
        - Top 5 predictions
        - Plant and disease names
    """
    # Check if model is loaded
    if model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Please ensure model file exists."
        }), 500
    
    if class_indices is None:
        return jsonify({
            "success": False,
            "error": "Class indices not loaded."
        }), 500
    
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided. Use 'image' field in form-data."
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "No image selected"
        }), 400
    
    try:
        # Read and preprocess image
        image = Image.open(BytesIO(file.read()))
        img_array = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top prediction
        top_idx = int(np.argmax(predictions))
        top_confidence = float(predictions[top_idx])
        top_class = class_indices.get(top_idx, f"Unknown_{top_idx}")
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions)[-5:][::-1]
        top_5_predictions = []
        for idx in top_5_indices:
            idx = int(idx)
            class_name = class_indices.get(idx, f"Unknown_{idx}")
            parsed = parse_class_name(class_name)
            top_5_predictions.append({
                "class": class_name,
                "plant": parsed["plant"],
                "disease": parsed["disease"],
                "confidence": float(predictions[idx])
            })
        
        # Parse top prediction
        parsed_top = parse_class_name(top_class)
        
        return jsonify({
            "success": True,
            "prediction": {
                "class": top_class,
                "plant": parsed_top["plant"],
                "disease": parsed_top["disease"],
                "confidence": top_confidence
            },
            "top_5_predictions": top_5_predictions
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error processing image: {str(e)}"
        }), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation."""
    return jsonify({
        "name": "Plant Disease Detection API",
        "version": "1.0.0",
        "description": "AI-powered plant disease detection from leaf images",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "GET /classes": "List all disease classes",
            "POST /predict": "Predict disease from image (multipart/form-data with 'image' field)"
        },
        "example": {
            "curl": "curl -X POST -F 'image=@leaf.jpg' http://localhost:5000/predict"
        }
    })


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    # Load model and classes on startup
    load_model_and_classes()
    
    print(f"\nStarting server on {HOST}:{PORT}")
    print(f"Health check: http://localhost:{PORT}/health")
    print(f"API docs: http://localhost:{PORT}/")
    
    app.run(host=HOST, port=PORT, debug=DEBUG)

