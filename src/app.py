"""
Flask web application for medical report analysis and disease prediction.
FIXED VERSION - Handles both standard CSV and text-format CSV uploads
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
from pathlib import Path
import io

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


def parse_csv_data(file):
    """
    FIXED: Parse CSV file - handles both standard CSV and text-format CSV.
    """
    try:
        # Read the file content
        file_content = file.read().decode('utf-8')
        file.seek(0)  # Reset file pointer
        
        print(f"DEBUG: File content (first 200 chars):\n{file_content[:200]}")
        
        # Check if it's text format (key: value) in CSV
        if ':' in file_content.split('\n')[0]:
            print("DEBUG: Detected text-format CSV (key: value)")
            # This is a text-format CSV like:
            # "glucose: 105"
            # "blood_pressure: 175"
            
            data = {}
            for line in file_content.split('\n'):
                line = line.strip().strip('"').strip("'")  # Remove quotes
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    try:
                        data[key] = float(value)
                    except ValueError:
                        continue
            
            print(f"DEBUG: Parsed text-format data: {data}")
            return data
        
        else:
            # Standard CSV format with columns
            print("DEBUG: Detected standard CSV format")
            df = pd.read_csv(file)
            
            print(f"DEBUG: CSV columns: {list(df.columns)}")
            print(f"DEBUG: CSV shape: {df.shape}")
            print(f"DEBUG: First row:\n{df.head(1)}")
            
            # Normalize column names (lowercase, replace spaces with underscores)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            print(f"DEBUG: Normalized columns: {list(df.columns)}")
            
            # Check if we have required columns
            required_found = sum(1 for col in FEATURE_COLUMNS if col in df.columns)
            
            if required_found >= 5:  # If we have at least 5 required columns
                # Get first data row
                data = df.iloc[0].to_dict()
                
                # Convert all values to float
                for key in list(data.keys()):
                    try:
                        data[key] = float(data[key])
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        del data[key]
                
                print(f"DEBUG: Parsed standard CSV data: {data}")
                return data
            else:
                raise ValueError(f"CSV missing required columns. Found: {list(df.columns)}")
    
    except Exception as e:
        print(f"ERROR parsing CSV: {e}")
        raise ValueError(f"Could not parse CSV file: {str(e)}")


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
    missing_features = []
    
    for col in FEATURE_COLUMNS:
        if col in data:
            features.append(data[col])
        else:
            # Use median values for missing features
            default_values = {
                'glucose': 95,
                'blood_pressure': 120,
                'heart_rate': 75,
                'hemoglobin': 14,
                'cholesterol': 180,
                'bmi': 23,
                'age': 40
            }
            features.append(default_values.get(col, 0))
            missing_features.append(col)
    
    if missing_features:
        print(f"WARNING: Missing features filled with defaults: {missing_features}")
    
    # Convert to DataFrame to maintain feature names
    feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    
    print(f"DEBUG: Features before scaling: {features}")
    
    # CRITICAL FIX: Scale the features using the same scaler from training
    if scaler is not None:
        scaled_features = scaler.transform(feature_df)
        print(f"DEBUG: Features after scaling: {scaled_features[0]}")
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
    FIXED to properly handle both CSV formats and text input.
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = None
        
        # Get input data
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            print(f"DEBUG: Received file: {file.filename}")
            
            if file.filename.endswith('.csv'):
                # FIXED: Use the new CSV parser that handles both formats
                data = parse_csv_data(file)
            else:
                return jsonify({'error': 'Unsupported file format. Please upload CSV'}), 400
        else:
            report_text = request.form.get('report_text', '')
            if not report_text:
                return jsonify({'error': 'No data provided'}), 400
            print(f"DEBUG: Received text: {report_text}")
            data = parse_report(report_text)
        
        if not data:
            return jsonify({'error': 'Could not parse report data'}), 400
        
        print(f"DEBUG: Final parsed data: {data}")
        
        # FIXED: Check healthy status with stricter logic
        if is_healthy(data):
            prediction = 'healthy'
            confidence = 0.95
            print("DEBUG: All values within healthy range -> healthy")
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
            print(f"DEBUG: Class labels={label_encoder.classes_}")
        
        # Prepare response
        response = {
            'prediction': prediction,
            'confidence': f"{confidence * 100:.1f}%",
            'data': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                    for k, v in data.items()},
            'message': f"Analysis complete: {prediction}"
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"ERROR in analyze: {e}")
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