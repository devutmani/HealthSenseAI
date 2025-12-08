import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def generate_diabetes_case():
    """Generate a diabetes case with elevated glucose and related markers."""
    return {
        'glucose': np.random.normal(180, 30),          # High glucose
        'blood_pressure': np.random.normal(140, 15),   # Often elevated
        'heart_rate': np.random.normal(80, 10),
        'hemoglobin': np.random.normal(14, 1.5),       
        'cholesterol': np.random.normal(220, 25),      # Often high
        'bmi': np.random.normal(30, 5),                # Often overweight
        'age': np.random.normal(55, 12),
        'disease': 'diabetes'
    }


def generate_heart_case():
    """Generate a heart disease case with cardiovascular issues."""
    return {
        'glucose': np.random.normal(110, 20),
        'blood_pressure': np.random.normal(150, 20),   # High BP
        'heart_rate': np.random.normal(95, 12),        # Elevated HR
        'hemoglobin': np.random.normal(14, 1.5),
        'cholesterol': np.random.normal(240, 30),      # High cholesterol
        'bmi': np.random.normal(28, 4),                # Overweight
        'age': np.random.normal(60, 10),
        'disease': 'heart'
    }


def generate_anemia_case():
    """Generate an anemia case with low hemoglobin."""
    return {
        'glucose': np.random.normal(95, 15),
        'blood_pressure': np.random.normal(105, 12),   # Often low
        'heart_rate': np.random.normal(85, 12),        # May be elevated
        'hemoglobin': np.random.normal(9, 1.5),        # Low hemoglobin
        'cholesterol': np.random.normal(170, 20),
        'bmi': np.random.normal(22, 3),
        'age': np.random.normal(45, 15),
        'disease': 'anemia'
    }


def generate_healthy_case():
    """Generate a healthy case with normal values."""
    return {
        'glucose': np.random.normal(95, 10),           # Normal glucose
        'blood_pressure': np.random.normal(115, 8),    # Normal BP
        'heart_rate': np.random.normal(75, 8),         # Normal HR
        'hemoglobin': np.random.normal(14.5, 1),       # Normal hemoglobin
        'cholesterol': np.random.normal(170, 15),      # Normal cholesterol
        'bmi': np.random.normal(22, 2),                # Normal BMI
        'age': np.random.normal(40, 15),
        'disease': 'healthy'
    }


def clip_values(data):
    """Ensure values are within realistic physiological ranges."""
    data['glucose'] = np.clip(data['glucose'], 50, 400)
    data['blood_pressure'] = np.clip(data['blood_pressure'], 70, 200)
    data['heart_rate'] = np.clip(data['heart_rate'], 40, 150)
    data['hemoglobin'] = np.clip(data['hemoglobin'], 5, 20)
    data['cholesterol'] = np.clip(data['cholesterol'], 100, 350)
    data['bmi'] = np.clip(data['bmi'], 15, 50)
    data['age'] = np.clip(data['age'], 18, 90)
    return data


def generate_dataset(n_samples=500, diabetes_ratio=0.25, heart_ratio=0.25, 
                    anemia_ratio=0.20, healthy_ratio=0.30):
    """
    FIXED: Generate balanced synthetic dataset with realistic disease patterns.
    """
    
    # Calculate number of samples per class
    n_diabetes = int(n_samples * diabetes_ratio)
    n_heart = int(n_samples * heart_ratio)
    n_anemia = int(n_samples * anemia_ratio)
    n_healthy = n_samples - n_diabetes - n_heart - n_anemia
    
    print(f"Generating {n_samples} samples:")
    print(f"  Diabetes: {n_diabetes}")
    print(f"  Heart: {n_heart}")
    print(f"  Anemia: {n_anemia}")
    print(f"  Healthy: {n_healthy}")
    
    # Generate cases
    data = []
    
    # Generate diabetes cases
    for _ in range(n_diabetes):
        data.append(clip_values(generate_diabetes_case()))
    
    # Generate heart disease cases
    for _ in range(n_heart):
        data.append(clip_values(generate_heart_case()))
    
    # Generate anemia cases
    for _ in range(n_anemia):
        data.append(clip_values(generate_anemia_case()))
    
    # Generate healthy cases
    for _ in range(n_healthy):
        data.append(clip_values(generate_healthy_case()))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Round values for realism
    df['glucose'] = df['glucose'].round(1)
    df['blood_pressure'] = df['blood_pressure'].round(0)
    df['heart_rate'] = df['heart_rate'].round(0)
    df['hemoglobin'] = df['hemoglobin'].round(1)
    df['cholesterol'] = df['cholesterol'].round(0)
    df['bmi'] = df['bmi'].round(1)
    df['age'] = df['age'].round(0)
    
    print(f"\n✓ Dataset generated successfully!")
    print(f"\nClass distribution:")
    print(df['disease'].value_counts())
    
    print(f"\nSample statistics:")
    print(df.groupby('disease').mean().round(2))
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic medical data for training'
    )
    parser.add_argument('--out', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--n', type=int, default=500,
                       help='Number of samples to generate (default: 500)')
    parser.add_argument('--diabetes', type=float, default=0.25,
                       help='Proportion of diabetes cases (default: 0.25)')
    parser.add_argument('--heart', type=float, default=0.25,
                       help='Proportion of heart disease cases (default: 0.25)')
    parser.add_argument('--anemia', type=float, default=0.20,
                       help='Proportion of anemia cases (default: 0.20)')
    
    args = parser.parse_args()
    
    # Calculate healthy ratio
    healthy_ratio = 1.0 - args.diabetes - args.heart - args.anemia
    
    if healthy_ratio < 0:
        raise ValueError("Disease ratios sum to more than 1.0!")
    
    # Generate dataset
    df = generate_dataset(
        n_samples=args.n,
        diabetes_ratio=args.diabetes,
        heart_ratio=args.heart,
        anemia_ratio=args.anemia,
        healthy_ratio=healthy_ratio
    )
    
    # Ensure output directory exists
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Dataset saved to: {output_path}")
    
    # Show a few samples
    print("\nFirst 5 samples:")
    print(df.head())


if __name__ == '__main__':
    main()