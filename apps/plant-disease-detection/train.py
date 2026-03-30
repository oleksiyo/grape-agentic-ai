"""
Train script for Plant Disease Detection model.

This script trains an Xception-based model on the PlantVillage dataset
using transfer learning with two-phase training:
1. Feature extraction (frozen base model)
2. Fine-tuning (unfrozen last layers)

Usage:
    python train.py

Output:
    - models/plant_disease_model.keras - trained model
    - models/class_indices.json - class name mappings
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ============================================================
# Configuration
# ============================================================
DATA_DIR = './data/plantvillage dataset/color'
MODELS_DIR = './models'
MODEL_PATH = os.path.join(MODELS_DIR, 'plant_disease_model.keras')
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, 'class_indices.json')

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10  # Feature extraction epochs
EPOCHS_PHASE2 = 10  # Fine-tuning epochs
LEARNING_RATE_PHASE1 = 0.001
LEARNING_RATE_PHASE2 = 1e-5
FINE_TUNE_LAYERS = 30  # Number of layers to unfreeze in phase 2

RANDOM_STATE = 42


def load_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load dataset from directory structure into a DataFrame.
    
    Args:
        data_dir: Path to dataset directory with class subfolders
        
    Returns:
        DataFrame with columns: image_path, plant, disease, class
    """
    print("Loading dataset...")
    rows = []
    
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        if "___" not in folder:
            continue
        
        plant, disease = folder.split("___")
        
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            if not os.path.isfile(img_path):
                continue
            
            rows.append({
                "image_path": img_path,
                "plant": plant,
                "disease": disease,
                "class": folder
            })
    
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} images from {len(df['class'].unique())} classes")
    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.5):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        df: DataFrame with image data
        test_size: Fraction of data for test+validation
        val_size: Fraction of test_size for validation
        
    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    print("Splitting dataset...")
    
    train_data, temp_data = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        stratify=df['class'],
        random_state=RANDOM_STATE
    )
    
    valid_data, test_data = train_test_split(
        temp_data,
        test_size=val_size,
        shuffle=True,
        stratify=temp_data['class'],
        random_state=RANDOM_STATE
    )
    
    print(f"Training set: {len(train_data)} images ({len(train_data)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(valid_data)} images ({len(valid_data)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_data)} images ({len(test_data)/len(df)*100:.1f}%)")
    
    return train_data, valid_data, test_data


def create_data_generators(train_data, valid_data, test_data):
    """
    Create data generators with augmentation for training.
    
    Returns:
        Tuple of (train_gen, valid_gen, test_gen, num_classes)
    """
    print("Creating data generators...")
    
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Validation/Test generator (only rescaling)
    valid_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_data,
        x_col='image_path',
        y_col='class',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        shuffle=True
    )
    
    valid_gen = valid_test_datagen.flow_from_dataframe(
        dataframe=valid_data,
        x_col='image_path',
        y_col='class',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        shuffle=False
    )
    
    test_gen = valid_test_datagen.flow_from_dataframe(
        dataframe=test_data,
        x_col='image_path',
        y_col='class',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        shuffle=False
    )
    
    num_classes = len(train_gen.class_indices)
    print(f"Number of classes: {num_classes}")
    
    return train_gen, valid_gen, test_gen, num_classes


def build_model(num_classes: int, input_shape: tuple = (224, 224, 3)):
    """
    Build Xception model with custom classification head.
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        
    Returns:
        Tuple of (model, base_model)
    """
    print("Building model...")
    
    # Load pre-trained Xception
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model for feature extraction
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print(f"Total parameters: {model.count_params():,}")
    
    return model, base_model


def train_phase1(model, train_gen, valid_gen):
    """
    Phase 1: Feature extraction with frozen base model.
    """
    print("\n" + "=" * 60)
    print("Phase 1: Feature Extraction")
    print("=" * 60)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_PHASE1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def train_phase2(model, base_model, train_gen, valid_gen):
    """
    Phase 2: Fine-tuning with unfrozen top layers.
    """
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning")
    print("=" * 60)
    
    # Unfreeze last N layers
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    
    trainable_params = sum([
        tf.keras.backend.count_params(w) 
        for w in model.trainable_weights
    ])
    print(f"Trainable parameters after unfreezing: {trainable_params:,}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_PHASE2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-8,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, test_gen):
    """
    Evaluate model on test set.
    """
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    results = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    
    return results


def save_class_indices(train_gen):
    """
    Save class indices mapping to JSON file.
    """
    class_indices = train_gen.class_indices
    # Invert: index -> class_name
    index_to_class = {v: k for k, v in class_indices.items()}
    
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(index_to_class, f, indent=2)
    
    print(f"Class indices saved to {CLASS_INDICES_PATH}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Plant Disease Detection - Model Training")
    print("=" * 60)
    
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"\nError: Dataset not found at {DATA_DIR}")
        print("Please download the PlantVillage dataset:")
        print("  kaggle datasets download -d emmarex/plantdisease")
        print("  unzip plantdisease.zip -d data/")
        return
    
    # Load and split dataset
    df = load_dataset(DATA_DIR)
    train_data, valid_data, test_data = split_dataset(df)
    
    # Create data generators
    train_gen, valid_gen, test_gen, num_classes = create_data_generators(
        train_data, valid_data, test_data
    )
    
    # Build model
    model, base_model = build_model(num_classes)
    
    # Phase 1: Feature extraction
    train_phase1(model, train_gen, valid_gen)
    
    # Phase 2: Fine-tuning
    train_phase2(model, base_model, train_gen, valid_gen)
    
    # Evaluate on test set
    evaluate_model(model, test_gen)
    
    # Save model (final save after evaluation)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Save class indices
    save_class_indices(train_gen)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

