from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
from pathlib import Path

app = Flask(__name__)

# Configuration
MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'rf_model.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'
LABEL_ENCODER_PATH = MODEL_DIR / 'label_encoder.pkl'

# Load model artifacts
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("✓ Model, scaler, and label encoder loaded successfully")
except Exception as e:
    print(f"✗ Error loading model artifacts: {e}")
    model = None
    scaler = None
    label_encoder = None

# FIXED: Define proper healthy ranges (more realistic thresholds)
HEALTHY_RANGES = {
    'glucose': (70, 120),           # Fasting glucose mg/dL
    'blood_pressure': (90, 130),    # Systolic BP mmHg
    'heart_rate': (60, 100),        # BPM
    'hemoglobin': (12.0, 17.0),     # g/dL
    'cholesterol': (125, 200),      # mg/dL
    'bmi': (18.5, 24.9),           # Body Mass Index
    'age': (0, 120)                # Just for validation
}

# FIXED: Define the exact feature order used during training
FEATURE_COLUMNS = ['glucose', 'blood_pressure', 'heart_rate', 'hemoglobin', 
                   'cholesterol', 'bmi', 'age']


def parse_report(report_text):
    """
    Parse medical report from text format.
    Accepts key:value pairs separated by newlines or commas.
    """
    data = {}
    
    # Split by newlines or commas
    lines = report_text.replace(',', '\n').split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            
            # Try to convert to float
            try:
                data[key] = float(value)
            except ValueError:
                continue
    
    return data


def is_healthy(data):
    """
    FIXED: More strict healthy determination.
    Returns True only if ALL values are within healthy ranges.
    """
    # Check if we have enough data
    if len(data) < 3:
        return False
    
    violations = []
    
    for key, value in data.items():
        if key in HEALTHY_RANGES:
            min_val, max_val = HEALTHY_RANGES[key]
            if value < min_val or value > max_val:
                violations.append(f"{key}: {value} (normal: {min_val}-{max_val})")
    
    # FIXED: Return False if there are ANY violations
    # This prevents false "healthy" predictions
    return len(violations) == 0


def prepare_features(data):
    """
    FIXED: Prepare features in the correct order and scale them properly.
    """
    # Create feature array with correct column order
    features = []
    for col in FEATURE_COLUMNS:
        features.append(data.get(col, 0))  # Use 0 as default if missing
    
    # Convert to DataFrame to maintain feature names
    feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    
    # CRITICAL FIX: Scale the features using the same scaler from training
    if scaler is not None:
        scaled_features = scaler.transform(feature_df)
        return scaled_features
    else:
        return feature_df.values


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze medical report and predict disease.
    FIXED to properly handle predictions.
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get input data
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                data = df.iloc[0].to_dict()
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
        else:
            report_text = request.form.get('report_text', '')
            if not report_text:
                return jsonify({'error': 'No data provided'}), 400
            data = parse_report(report_text)
        
        if not data:
            return jsonify({'error': 'Could not parse report data'}), 400
        
        # FIXED: Check healthy status with stricter logic
        if is_healthy(data):
            prediction = 'healthy'
            confidence = 0.95
        else:
            # FIXED: Prepare features correctly and make prediction
            features = prepare_features(data)
            
            # Get prediction and probabilities
            pred_encoded = model.predict(features)[0]
            pred_proba = model.predict_proba(features)[0]
            
            # FIXED: Decode the prediction properly
            prediction = label_encoder.inverse_transform([pred_encoded])[0]
            confidence = float(max(pred_proba))
            
            print(f"DEBUG: Prediction={prediction}, Confidence={confidence}")
            print(f"DEBUG: All probabilities={pred_proba}")
        
        # Prepare response
        response = {
            'prediction': prediction,
            'confidence': f"{confidence * 100:.1f}%",
            'data': data,
            'message': f"Analysis complete: {prediction}"
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in analyze: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'label_encoder_loaded': label_encoder is not None
    }
    return jsonify(status)


if __name__ == '__main__':
    # Ensure model directory exists
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)