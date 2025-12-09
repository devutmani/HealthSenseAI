from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
from pathlib import Path
import io

app = Flask(__name__)

MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'rf_model.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'
LABEL_ENCODER_PATH = MODEL_DIR / 'label_encoder.pkl'

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

HEALTHY_RANGES = {
    'glucose': (70, 120),        
    'blood_pressure': (90, 130), 
    'heart_rate': (60, 100),     
    'hemoglobin': (12.0, 17.0),  
    'cholesterol': (125, 200),   
    'bmi': (18.5, 24.9),         
    'age': (0, 120)              
}

FEATURE_COLUMNS = ['glucose', 'blood_pressure', 'heart_rate', 'hemoglobin', 
                   'cholesterol', 'bmi', 'age']


def parse_report(report_text):

    data = {}
    lines = report_text.replace(',', '\n').split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            
            try:
                data[key] = float(value)
            except ValueError:
                continue
    
    return data


def parse_csv_data(file):
    try:
        file_content = file.read().decode('utf-8')
        file.seek(0)
        
        print(f"DEBUG: File content (first 200 chars):\n{file_content[:200]}")
        
        if ':' in file_content.split('\n')[0]:
            print("DEBUG: Detected text-format CSV (key: value)")
            
            data = {}
            for line in file_content.split('\n'):
                line = line.strip().strip('"').strip("'")  
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
            print("DEBUG: Detected standard CSV format")
            df = pd.read_csv(file)
            
            print(f"DEBUG: CSV columns: {list(df.columns)}")
            print(f"DEBUG: CSV shape: {df.shape}")
            print(f"DEBUG: First row:\n{df.head(1)}")
            
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            print(f"DEBUG: Normalized columns: {list(df.columns)}")
            
            required_found = sum(1 for col in FEATURE_COLUMNS if col in df.columns)
            
            if required_found >= 5: 
                data = df.iloc[0].to_dict()
                
                for key in list(data.keys()):
                    try:
                        data[key] = float(data[key])
                    except (ValueError, TypeError):
                        del data[key]
                
                print(f"DEBUG: Parsed standard CSV data: {data}")
                return data
            else:
                raise ValueError(f"CSV missing required columns. Found: {list(df.columns)}")
    
    except Exception as e:
        print(f"ERROR parsing CSV: {e}")
        raise ValueError(f"Could not parse CSV file: {str(e)}")


def is_healthy(data):
    if len(data) < 3:
        return False
    
    violations = []
    
    for key, value in data.items():
        if key in HEALTHY_RANGES:
            min_val, max_val = HEALTHY_RANGES[key]
            if value < min_val or value > max_val:
                violations.append(f"{key}: {value} (normal: {min_val}-{max_val})")
    
    return len(violations) == 0


def prepare_features(data):
    features = []
    missing_features = []
    
    for col in FEATURE_COLUMNS:
        if col in data:
            features.append(data[col])
        else:
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
    
    feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    
    print(f"DEBUG: Features before scaling: {features}")
    
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
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = None
        
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            print(f"DEBUG: Received file: {file.filename}")
            
            if file.filename.endswith('.csv'):
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
        
        if is_healthy(data):
            prediction = 'healthy'
            confidence = 0.95
            print("DEBUG: All values within healthy range -> healthy")
        else:
            features = prepare_features(data)
            
            pred_encoded = model.predict(features)[0]
            pred_proba = model.predict_proba(features)[0]
            
            prediction = label_encoder.inverse_transform([pred_encoded])[0]
            confidence = float(max(pred_proba))
            
            print(f"DEBUG: Prediction={prediction}, Confidence={confidence}")
            print(f"DEBUG: All probabilities={pred_proba}")
            print(f"DEBUG: Class labels={label_encoder.classes_}")
    
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
    status = {
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'label_encoder_loaded': label_encoder is not None
    }
    return jsonify(status)


if __name__ == '__main__':

    MODEL_DIR.mkdir(exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)