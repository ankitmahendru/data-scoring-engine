"""
Script to run Data Quality Engine on India House Rent Prediction dataset
Dataset: https://www.kaggle.com/datasets/pranavshinde36/india-house-rent-prediction/data

Setup Instructions:
1. Download the dataset from Kaggle (House_Rent_Dataset.csv)
2. Place it in the same directory as this script
3. Run: python run_kaggle_dataset.py

OR use Kaggle API:
1. Install: pip install kaggle
2. Setup API credentials: https://github.com/Kaggle/kaggle-api#api-credentials
3. Run this script - it will auto-download
"""

import pandas as pd
import os
from pathlib import Path

# Import the engine (assumes data_quality_engine.py is in same directory)
from data_quality_engine import DataQualityEngine


def download_dataset():
    """Download dataset using Kaggle API if not present."""
    try:
        import kaggle
        print("Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            'pranavshinde36/india-house-rent-prediction',
            path='.',
            unzip=True
        )
        print("Download complete!")
        return True
    except ImportError:
        print("Kaggle API not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from Kaggle and place CSV in this directory.")
        return False


def load_dataset():
    """Load the dataset from local file."""
    csv_file = 'House_Rent_Dataset.csv'
    
    if not os.path.exists(csv_file):
        print(f"Dataset not found: {csv_file}")
        print("Attempting to download...")
        download_dataset()
    
    if not os.path.exists(csv_file):
        print("\n" + "="*70)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*70)
        print("1. Go to: https://www.kaggle.com/datasets/pranavshinde36/india-house-rent-prediction/data")
        print("2. Click 'Download' button")
        print("3. Extract House_Rent_Dataset.csv to this directory")
        print("4. Run this script again")
        print("="*70)
        return None
    
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def analyze_dataset():
    """Main analysis function."""
    
    # Load data
    df = load_dataset()
    if df is None:
        return
    
    # Display dataset info
    print("\n" + "="*70)
    print("DATASET PREVIEW")
    print("="*70)
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head(3))
    print(f"\nData types:")
    print(df.dtypes)
    
    # Run quality assessment
    print("\n" + "="*70)
    print("RUNNING DATA QUALITY ASSESSMENT")
    print("="*70)
    print("Target column: 'Rent' (the price to predict)")
    print("This may take 10-30 seconds for this dataset size...")
    print()
    
    engine = DataQualityEngine()
    
    # The target variable in this dataset is 'Rent'
    result = engine.assess(df, target_col='Rent')
    
    # Print the full report
    engine.print_report(result)
    
    # Additional insights specific to this dataset
    print("\n" + "="*70)
    print("DOMAIN-SPECIFIC INSIGHTS")
    print("="*70)
    
    print("\n🏠 Housing Dataset Context:")
    print(f"  • Dataset size: {len(df):,} rental listings")
    print(f"  • Price range: ₹{df['Rent'].min():,.0f} - ₹{df['Rent'].max():,.0f}")
    print(f"  • Median rent: ₹{df['Rent'].median():,.0f}")
    
    if 'City' in df.columns:
        print(f"  • Cities covered: {df['City'].nunique()}")
        print(f"  • Top cities: {df['City'].value_counts().head(3).to_dict()}")
    
    if 'BHK' in df.columns:
        print(f"  • BHK types: {sorted(df['BHK'].unique())}")
    
    if 'Furnishing Status' in df.columns:
        print(f"  • Furnishing: {df['Furnishing Status'].value_counts().to_dict()}")
    
    # Save results to file
    output_file = 'dq_assessment_results.txt'
    with open(output_file, 'w') as f:
        f.write(f"Data Quality Assessment - India House Rent Dataset\n")
        f.write(f"Overall Score: {result['score']}/100\n")
        f.write(f"Readiness: {result['readiness']}\n\n")
        f.write(f"Signals:\n")
        for signal in result['signals']:
            f.write(f"  {signal.name}: {signal.score}/100 ({signal.severity})\n")
            f.write(f"  Message: {signal.message}\n\n")
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Return structured results for further analysis
    return result


if __name__ == "__main__":
    print("="*70)
    print("DATA QUALITY ENGINE - KAGGLE HOUSING DATASET ANALYSIS")
    print("="*70)
    print()
    
    result = analyze_dataset()
    
    if result:
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nFinal Assessment: {result['readiness']}")
        print(f"Quality Score: {result['score']}/100")
        print("\nNext Steps:")
        if result['score'] >= 85:
            print("  ✓ Dataset is ready for modeling")
            print("  ✓ Proceed with feature engineering and model selection")
        elif result['score'] >= 70:
            print("  → Review warnings above and address if possible")
            print("  → Acceptable to proceed with caution")
        else:
            print("  ⚠ Address critical issues before modeling")
            print("  ⚠ Review the recommendations section above")