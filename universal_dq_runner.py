"""
Universal Data Quality Runner for ANY Kaggle Dataset

Usage:
    python universal_dq_runner.py <dataset-slug> [options]

Examples:
    python universal_dq_runner.py pranavshinde36/india-house-rent-prediction
    python universal_dq_runner.py -d pranavshinde36/india-house-rent-prediction -t Rent
    python universal_dq_runner.py -f local_data.csv -t price
    python universal_dq_runner.py -d uciml/iris -t species --auto-detect

Requirements:
    pip install pandas numpy scikit-learn kaggle

Setup Kaggle API (one-time setup that always takes longer than expected):
    1. Go to kaggle.com → Account → Create New API Token
    2. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)
    3. chmod 600 ~/.kaggle/kaggle.json (Linux/Mac only)
    
Yes, you need to do ALL of these steps. No, there's no shortcut.
"""

import pandas as pd
import os
import sys
import argparse
import zipfile
from pathlib import Path
from typing import Optional, List
import json

# Import the engine
try:
    from data_quality_engine import DataQualityEngine
except ImportError:
    print("ERROR: data_quality_engine.py not found in current directory")
    print("Make sure data_quality_engine.py is in the same folder as this script")
    sys.exit(1)


class UniversalDatasetRunner:
    """Handles any Kaggle dataset or local CSV file.
    
    The swiss army knife of data quality checking.
    """
    
    def __init__(self, output_dir: str = 'dq_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.engine = DataQualityEngine()
    
    def download_kaggle_dataset(self, dataset_slug: str) -> Optional[Path]:
        """Download dataset from Kaggle using API.
        
        This will fail at least once. Probably because of credentials.
        """
        try:
            import kaggle
        except ImportError:
            print("ERROR: Kaggle API not installed")
            print("Install with: pip install kaggle")
            return None
        
        try:
            download_path = self.output_dir / 'kaggle_downloads' / dataset_slug.replace('/', '_')
            download_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Downloading dataset: {dataset_slug}")
            print(f"Destination: {download_path}")
            
            kaggle.api.dataset_download_files(
                dataset_slug,
                path=str(download_path),
                unzip=True,
                quiet=False
            )
            
            print("✓ Download complete!")
            return download_path
            
        except Exception as e:
            print(f"ERROR: Download failed - {e}")
            print("\nTroubleshooting:")
            print("1. Check dataset slug format: 'username/dataset-name'")
            print("2. Verify Kaggle API credentials are set up")
            print("3. Ensure dataset exists and is public")
            return None
    
    def find_csv_files(self, directory: Path) -> List[Path]:
        """Find all CSV files in directory.
        
        Because datasets never have just ONE csv file, do they?
        """
        csv_files = list(directory.glob('*.csv'))
        
        # Also check subdirectories (sigh)
        for subdir in directory.iterdir():
            if subdir.is_dir():
                csv_files.extend(subdir.glob('*.csv'))
        
        return csv_files
    
    def auto_detect_target(self, df: pd.DataFrame) -> Optional[str]:
        """Attempt to automatically detect the target column.
        
        Works 60% of the time, every time.
        """
        
        # Common target column name patterns (based on years of pain)
        target_patterns = [
            'target', 'label', 'class', 'output', 'y',
            'price', 'cost', 'amount', 'value', 'salary', 'income',
            'survived', 'outcome', 'result', 'prediction',
            'category', 'type', 'status', 'churn', 'fraud'
        ]
        
        # Check for exact matches (case-insensitive)
        for col in df.columns:
            if col.lower() in target_patterns:
                return col
        
        # Check for partial matches
        for col in df.columns:
            for pattern in target_patterns:
                if pattern in col.lower():
                    return col
        
        # Heuristic: Last column is often the target
        # But only if its numeric or has few unique values
        last_col = df.columns[-1]
        if df[last_col].dtype in ['int64', 'float64', 'object']:
            if df[last_col].nunique() < len(df) * 0.5:  # Not an ID column
                print(f"HEURISTIC: Guessing last column '{last_col}' as target")
                return last_col
        
        return None
    
    def smart_load_csv(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """Load CSV with intelligent error handling.
        
        Because CSV files are the wild west of data formats.
        """
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        separators = [',', ';', '\t', '|']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding, sep=sep, low_memory=False)
                    
                    # Validate its actually a proper CSV
                    if len(df.columns) > 1 and len(df) > 0:
                        print(f"✓ Loaded with encoding={encoding}, separator='{sep}'")
                        return df
                except Exception:
                    continue  # try next combination
        
        print(f"ERROR: Could not load {csv_path.name} with any encoding/separator combo")
        return None
    
    def interactive_target_selection(self, df: pd.DataFrame) -> Optional[str]:
        """Let user interactively select target column.
        
        For when auto-detection fails (which is often).
        """
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            nunique = df[col].nunique()
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
            print(f"  {i:2d}. {col:30s} | {str(dtype):10s} | {nunique:5d} unique | Sample: {sample}")
        
        print("\nOptions:")
        print("  - Enter column number (1-{})".format(len(df.columns)))
        print("  - Enter column name directly")
        print("  - Press Enter to skip target specification")
        
        user_input = input("\nSelect target column: ").strip()
        
        if not user_input:
            return None
        
        # Try as number
        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(df.columns):
                return df.columns[idx]
        except ValueError:
            pass
        
        # Try as column name
        if user_input in df.columns:
            return user_input
        
        # Fuzzy match (for when users cant spell)
        for col in df.columns:
            if user_input.lower() in col.lower():
                confirm = input(f"Did you mean '{col}'? (y/n): ").strip().lower()
                if confirm == 'y':
                    return col
        
        print("Invalid selection, skipping target specification")
        return None
    
    def analyze_csv(self, csv_path: Path, target_col: Optional[str] = None, 
                    auto_detect: bool = False, interactive: bool = False) -> dict:
        """Analyze a single CSV file.
        
        The workhorse function that does all the heavy lifting.
        """
        
        print("\n" + "="*80)
        print(f"ANALYZING: {csv_path.name}")
        print("="*80)
        
        # Load data
        df = self.smart_load_csv(csv_path)
        if df is None:
            return {'error': 'Failed to load CSV'}
        
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Handle target column selection
        if target_col and target_col not in df.columns:
            print(f"WARNING: Specified target '{target_col}' not found in dataset")
            target_col = None
        
        if target_col is None and auto_detect:
            target_col = self.auto_detect_target(df)
            if target_col:
                print(f"AUTO-DETECTED target column: '{target_col}'")
        
        if target_col is None and interactive:
            target_col = self.interactive_target_selection(df)
        
        if target_col:
            print(f"Using target column: '{target_col}'")
        else:
            print("No target column specified (some checks will be skipped)")
        
        # Run assessment (the moment of truth)
        print("\nRunning quality assessment...")
        result = self.engine.assess(df, target_col=target_col)
        
        # Print report
        self.engine.print_report(result)
        
        # Save results (for posterity)
        output_file = self.output_dir / f"{csv_path.stem}_dq_report.txt"
        self.save_results(result, output_file, csv_path.name, df)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        return result
    
    def save_results(self, result: dict, output_file: Path, 
                     dataset_name: str, df: pd.DataFrame):
        """Save detailed results to file.
        
        Because terminal output disappears and your manager needs proof.
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DATA QUALITY ASSESSMENT REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Shape: {result['dataset_shape'][0]:,} rows × {result['dataset_shape'][1]} columns\n")
            f.write(f"Target: {result['target_column'] or 'Not specified'}\n\n")
            
            f.write(f"OVERALL SCORE: {result['score']}/100\n")
            f.write(f"READINESS: {result['readiness']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("DETAILED FINDINGS\n")
            f.write("="*80 + "\n\n")
            
            for signal in result['signals']:
                f.write(f"[{signal.name.upper()}] Score: {signal.score}/100 ({signal.severity})\n")
                f.write(f"Message: {signal.message}\n")
                
                if signal.details:
                    f.write("Details:\n")
                    for key, value in signal.details.items():
                        if isinstance(value, dict) and len(value) <= 5:
                            f.write(f"  {key}: {value}\n")
                        elif isinstance(value, list) and len(value) <= 5:
                            f.write(f"  {key}: {value}\n")
                        elif isinstance(value, (int, float)):
                            f.write(f"  {key}: {value}\n")
                
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("COLUMN SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            for col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                f.write(f"{col}:\n")
                f.write(f"  Type: {df[col].dtype}\n")
                f.write(f"  Missing: {missing_pct:.1f}%\n")
                f.write(f"  Unique: {df[col].nunique()}\n\n")
    
    def run(self, dataset_slug: Optional[str] = None, csv_file: Optional[str] = None,
            target_col: Optional[str] = None, auto_detect: bool = False, 
            interactive: bool = False):
        """Main entry point.
        
        Where the magic happens (or crashes, one of the two).
        """
        
        if csv_file:
            # Analyze local file
            csv_path = Path(csv_file)
            if not csv_path.exists():
                print(f"ERROR: File not found: {csv_file}")
                return
            
            self.analyze_csv(csv_path, target_col, auto_detect, interactive)
        
        elif dataset_slug:
            # Download and analyze Kaggle dataset
            download_path = self.download_kaggle_dataset(dataset_slug)
            if not download_path:
                return
            
            csv_files = self.find_csv_files(download_path)
            
            if not csv_files:
                print(f"ERROR: No CSV files found in {download_path}")
                return
            
            print(f"\nFound {len(csv_files)} CSV file(s)")
            
            for csv_file in csv_files:
                self.analyze_csv(csv_file, target_col, auto_detect, interactive)
                if len(csv_files) > 1:
                    print("\n" + "="*80 + "\n")
        
        else:
            print("ERROR: Must specify either --dataset or --file")
            print("Use --help for usage instructions")


def main():
    parser = argparse.ArgumentParser(
        description='Universal Data Quality Runner for Kaggle Datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Kaggle dataset with auto-detection
  python universal_dq_runner.py -d pranavshinde36/india-house-rent-prediction --auto-detect
  
  # Kaggle dataset with specified target
  python universal_dq_runner.py -d uciml/iris -t species
  
  # Local CSV file with interactive target selection
  python universal_dq_runner.py -f data.csv --interactive
  
  # Multiple CSVs in downloaded dataset
  python universal_dq_runner.py -d dataset-with-multiple-csvs --auto-detect
        """
    )
    
    parser.add_argument('-d', '--dataset', type=str,
                       help='Kaggle dataset slug (e.g., username/dataset-name)')
    
    parser.add_argument('-f', '--file', type=str,
                       help='Local CSV file path')
    
    parser.add_argument('-t', '--target', type=str,
                       help='Target column name for supervised learning')
    
    parser.add_argument('--auto-detect', action='store_true',
                       help='Automatically detect target column')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Interactively select target column')
    
    parser.add_argument('-o', '--output', type=str, default='dq_results',
                       help='Output directory for results (default: dq_results)')
    
    parser.add_argument('dataset_positional', nargs='?', type=str,
                       help='Kaggle dataset slug (positional argument)')
    
    args = parser.parse_args()
    
    # Handle positional argument (for lazy people like me)
    if args.dataset_positional and not args.dataset and not args.file:
        args.dataset = args.dataset_positional
    
    # Validate inputs
    if not args.dataset and not args.file:
        parser.print_help()
        sys.exit(1)
    
    # Run analysis (fingers crossed)
    runner = UniversalDatasetRunner(output_dir=args.output)
    runner.run(
        dataset_slug=args.dataset,
        csv_file=args.file,
        target_col=args.target,
        auto_detect=args.auto_detect,
        interactive=args.interactive
    )


if __name__ == "__main__":
    main()