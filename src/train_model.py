import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import argparse
from pathlib import Path
import json

FEATURE_COLUMNS = ['glucose', 'blood_pressure', 'heart_rate', 'hemoglobin', 
                   'cholesterol', 'bmi', 'age']
TARGET_COLUMN = 'disease'


def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Columns: {list(df.columns)}")
    return df

def preprocess_data(df):
    
    print("\nPreprocessing data...")
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")
    
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    X = X.fillna(X.median())
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target distribution:\n{y.value_counts()}")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"✓ Label encoding: {dict(zip(label_encoder.classes_, 
                                        label_encoder.transform(label_encoder.classes_)))}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✓ Feature scaling complete")
    print(f"  Mean: {X_scaled.mean(axis=0)}")
    print(f"  Std: {X_scaled.std(axis=0)}")
    
    return X_scaled, y_encoded, scaler, label_encoder


def train_model(X_train, y_train):

    print("\nTraining RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=200,          
        max_depth=10,              
        min_samples_split=10,      
        min_samples_leaf=5,        
        max_features='sqrt',       
        class_weight='balanced',   
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("✓ Model training complete")
    
    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Accuracy: {accuracy:.4f}")
    
    target_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nFeature Importance:")
    for col, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
        print(f"  {col}: {importance:.4f}")
    
    return accuracy


def save_artifacts(model, scaler, label_encoder, output_dir):
    """Save model and preprocessing artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving artifacts to {output_dir}...")
    
    model_path = output_dir / 'rf_model.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    scaler_path = output_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}")
    
    encoder_path = output_dir / 'label_encoder.pkl'
    joblib.dump(label_encoder, encoder_path)
    print(f"✓ Label encoder saved: {encoder_path}")
    
    metadata = {
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN,
        'label_mapping': dict(zip(label_encoder.classes_, 
                                 label_encoder.transform(label_encoder.classes_).tolist()))
    }
    
    metadata_path = output_dir / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Train disease prediction model')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to training data CSV')
    parser.add_argument('--out', type=str, default='models',
                       help='Output directory for model artifacts')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    df = load_data(args.data)
    
    X, y, scaler, label_encoder = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\n✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    model = train_model(X_train, y_train)
    
    evaluate_model(model, X_test, y_test, label_encoder)
    
    save_artifacts(model, scaler, label_encoder, args.out)
    
    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()