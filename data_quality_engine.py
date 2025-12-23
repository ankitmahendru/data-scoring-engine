"""
Data Quality Scoring Engine
A pragmatic system for assessing ML dataset readiness through quality signals.

Because apparently we need to check data BEFORE feeding it to models.
Who knew? (everyone. everyone knew.)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


@dataclass
class QualitySignal:
    """Container for individual quality assessment results."""
    name: str
    score: float  # 0-100, higher is better (obviously)
    severity: str  # 'critical', 'warning', 'info'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class MissingValueAnalyzer:
    """Detect problematic missing value patterns.
    
    Spoiler alert: your data has missing values. It always does.
    """
    
    def analyze(self, df: pd.DataFrame) -> QualitySignal:
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100
        
        # Check for systematic missingness (the bad kind)
        col_missing = df.isnull().sum() / len(df) * 100
        high_missing_cols = col_missing[col_missing > 50].index.tolist()
        
        # Check for rows that are basically empty
        row_missing = df.isnull().sum(axis=1) / df.shape[1] * 100
        problematic_rows = (row_missing > 70).sum()
        
        # Scoring logic - totally arbitary numbers that work in practice
        if missing_pct > 30:
            score = max(0, 100 - missing_pct * 2)
            severity = 'critical'
            msg = f"Severe data sparsity: {missing_pct:.1f}% missing values"
        elif missing_pct > 10:
            score = max(50, 100 - missing_pct * 3)
            severity = 'warning'
            msg = f"Moderate missingness: {missing_pct:.1f}% missing values"
        else:
            score = 100 - missing_pct * 2
            severity = 'info'
            msg = f"Acceptable missingness: {missing_pct:.1f}% missing values"
        
        details = {
            'total_missing_pct': missing_pct,
            'high_missing_columns': high_missing_cols,
            'problematic_rows': problematic_rows
        }
        
        return QualitySignal('missing_values', score, severity, msg, details)


class OutlierAnalyzer:
    """Detect anomalous density and data distribution issues.
    
    Fun fact: sometimes your outliers aren't outliers, they're just bad data.
    Other times they're billionares. Hard to tell without context.
    """
    
    def analyze(self, df: pd.DataFrame) -> QualitySignal:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return QualitySignal('outliers', 100, 'info', 
                               'No numeric features to analyze', {})
        
        outlier_counts = {}
        contamination_rates = []
        
        for col in numeric_cols:
            clean_data = df[col].dropna()
            if len(clean_data) < 10:
                continue  # not enough data to be statistically annoying
            
            # IQR method - the classic approach your stats prof taught you
            Q1, Q3 = clean_data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((clean_data < lower_bound) | (clean_data > upper_bound)).sum()
            outlier_pct = (outliers / len(clean_data)) * 100
            
            if outlier_pct > 5:
                outlier_counts[col] = outlier_pct
                contamination_rates.append(outlier_pct)
        
        # Multivariate outlier detection (fancy stuff)
        if len(numeric_cols) >= 3:
            clean_df = df[numeric_cols].dropna()
            if len(clean_df) >= 20:
                try:
                    # isolation forest: because trees solve everything
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    predictions = iso_forest.fit_predict(clean_df)
                    multivar_outliers = (predictions == -1).sum()
                    multivar_pct = (multivar_outliers / len(clean_df)) * 100
                    contamination_rates.append(multivar_pct)
                except:
                    pass  # silently fail like a true production system
        
        # Scoring - more arbitrary thresholds!
        avg_contamination = np.mean(contamination_rates) if contamination_rates else 0
        
        if avg_contamination > 15:
            score = max(30, 100 - avg_contamination * 3)
            severity = 'critical'
            msg = f"High outlier density: {avg_contamination:.1f}% anomalies detected"
        elif avg_contamination > 5:
            score = max(60, 100 - avg_contamination * 4)
            severity = 'warning'
            msg = f"Moderate outliers: {avg_contamination:.1f}% anomalies detected"
        else:
            score = 95
            severity = 'info'
            msg = f"Low outlier rate: {avg_contamination:.1f}% anomalies"
        
        details = {
            'columns_with_outliers': outlier_counts,
            'avg_contamination_pct': avg_contamination
        }
        
        return QualitySignal('outliers', score, severity, msg, details)


class LeakageDetector:
    """Detect potential feature leakage indicators.
    
    AKA the "why is my model TOO good" detector.
    If your accuracy is 99.9%, you probably have leakage. Sorry.
    """
    
    def analyze(self, df: pd.DataFrame, target_col: str = None) -> QualitySignal:
        if target_col is None or target_col not in df.columns:
            return QualitySignal('leakage', 100, 'info',
                               'No target specified, leakage check skipped', {})
        
        warnings_list = []
        severity = 'info'
        
        # Check for perfect correlations (the smoking gun)
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col in numeric_df.columns and len(numeric_df.columns) > 1:
            target = numeric_df[target_col]
            corr_with_target = numeric_df.drop(columns=[target_col]).corrwith(target).abs()
            
            perfect_corr = corr_with_target[corr_with_target > 0.99].index.tolist()
            high_corr = corr_with_target[(corr_with_target > 0.95) & (corr_with_target <= 0.99)].index.tolist()
            
            if perfect_corr:
                warnings_list.append(f"Perfect correlation with target: {perfect_corr}")
                severity = 'critical'
            
            if high_corr:
                warnings_list.append(f"Suspiciously high correlation: {high_corr}")
                if severity != 'critical':
                    severity = 'warning'
        
        # Check for duplicate columns (why do people do this?)
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                if df[col1].equals(df[col2]):
                    warnings_list.append(f"Duplicate columns: '{col1}' and '{col2}'")
                    severity = 'warning'
        
        # Check for ID-like columns that shouldnt be features
        id_patterns = ['id', 'index', 'key', 'uuid', 'guid']
        for col in df.columns:
            if col.lower() != target_col and any(pattern in col.lower() for pattern in id_patterns):
                if df[col].nunique() / len(df) > 0.95:
                    warnings_list.append(f"Potential ID column as feature: '{col}'")
                    if severity == 'info':
                        severity = 'warning'
        
        # Scoring
        if severity == 'critical':
            score = 20
            msg = "CRITICAL: Strong leakage indicators detected"
        elif severity == 'warning':
            score = 60
            msg = "WARNING: Potential leakage patterns found"
        else:
            score = 100
            msg = "No obvious leakage indicators"
        
        details = {'warnings': warnings_list}
        return QualitySignal('leakage', score, severity, msg, details)


class LabelNoiseEstimator:
    """Estimate label quality and consistency.
    
    Because labels are usually hand-annotated by tired humans or buggy scripts.
    """
    
    def analyze(self, df: pd.DataFrame, target_col: str = None) -> QualitySignal:
        if target_col is None or target_col not in df.columns:
            return QualitySignal('label_noise', 100, 'info',
                               'No target specified, label noise check skipped', {})
        
        target = df[target_col].dropna()
        
        # Check class imbalance (for classification tasks)
        if target.dtype == 'object' or target.nunique() < 20:
            value_counts = target.value_counts()
            imbalance_ratio = value_counts.max() / value_counts.min() if len(value_counts) > 1 else 1
            
            if imbalance_ratio > 100:
                score = 40
                severity = 'critical'
                msg = f"Severe class imbalance: {imbalance_ratio:.0f}:1 ratio"
            elif imbalance_ratio > 10:
                score = 70
                severity = 'warning'
                msg = f"Moderate class imbalance: {imbalance_ratio:.1f}:1 ratio"
            else:
                score = 95
                severity = 'info'
                msg = f"Balanced classes: {imbalance_ratio:.1f}:1 ratio"
            
            details = {
                'class_distribution': value_counts.to_dict(),
                'imbalance_ratio': imbalance_ratio
            }
        else:
            # For regression, check for label consistency
            std_dev = target.std()
            mean_val = target.mean()
            cv = (std_dev / mean_val) if mean_val != 0 else float('inf')
            
            score = 90
            severity = 'info'
            msg = f"Continuous target with CV={cv:.2f}"
            details = {'coefficient_variation': cv}
        
        return QualitySignal('label_noise', score, severity, msg, details)


class DataQualityEngine:
    """Main orchestrator for data quality assessment.
    
    The thing that actually runs all the checks.
    Think of it as a health checkup for your data.
    """
    
    def __init__(self):
        self.analyzers = {
            'missing': MissingValueAnalyzer(),
            'outliers': OutlierAnalyzer(),
            'leakage': LeakageDetector(),
            'labels': LabelNoiseEstimator()
        }
    
    def assess(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """Run all quality checks and compute aggregate score.
        
        This is the main function you actually care about.
        """
        
        # Basic validation
        if df.empty:
            return self._empty_dataset_result()
        
        # Run all analyzers (the fun part)
        signals = []
        signals.append(self.analyzers['missing'].analyze(df))
        signals.append(self.analyzers['outliers'].analyze(df))
        signals.append(self.analyzers['leakage'].analyze(df, target_col))
        signals.append(self.analyzers['labels'].analyze(df, target_col))
        
        # Compute weighted aggregate score (weights totally made up but work well)
        weights = {
            'missing_values': 0.30,
            'outliers': 0.25,
            'leakage': 0.30,
            'label_noise': 0.15
        }
        
        weighted_sum = sum(s.score * weights[s.name] for s in signals)
        
        final_score = round(weighted_sum, 1)
        
        # Generate readiness assessment
        if final_score >= 85:
            readiness = "READY - Dataset is production-grade"
        elif final_score >= 70:
            readiness = "CAUTION - Addressable issues present"
        elif final_score >= 50:
            readiness = "NEEDS WORK - Significant quality gaps"
        else:
            readiness = "NOT READY - Critical issues must be resolved"
        
        return {
            'score': final_score,
            'readiness': readiness,
            'signals': signals,
            'dataset_shape': df.shape,
            'target_column': target_col
        }
    
    def _empty_dataset_result(self) -> Dict[str, Any]:
        """Handle the edge case where someone passes an empty dataframe.
        
        Yes, this has happened in production. Multiple times.
        """
        return {
            'score': 0,
            'readiness': 'NOT READY - Empty dataset',
            'signals': [],
            'dataset_shape': (0, 0),
            'target_column': None
        }
    
    def print_report(self, result: Dict[str, Any]):
        """Generate human-readable report.
        
        Makes the output look fancy for stakeholders.
        """
        print("=" * 70)
        print("DATA QUALITY ASSESSMENT REPORT")
        print("=" * 70)
        print(f"\nDataset Shape: {result['dataset_shape'][0]} rows × {result['dataset_shape'][1]} columns")
        if result['target_column']:
            print(f"Target Column: '{result['target_column']}'")
        print(f"\n{'='*70}")
        print(f"OVERALL SCORE: {result['score']}/100")
        print(f"READINESS: {result['readiness']}")
        print(f"{'='*70}\n")
        
        # Group by severity (so you know what to panic about first)
        critical = [s for s in result['signals'] if s.severity == 'critical']
        warnings = [s for s in result['signals'] if s.severity == 'warning']
        info = [s for s in result['signals'] if s.severity == 'info']
        
        if critical:
            print("🚨 CRITICAL ISSUES (Must Fix)")
            print("-" * 70)
            for signal in critical:
                print(f"  [{signal.name.upper()}] {signal.message}")
                self._print_details(signal.details)
            print()
        
        if warnings:
            print("⚠️  WARNINGS (Should Address)")
            print("-" * 70)
            for signal in warnings:
                print(f"  [{signal.name.upper()}] {signal.message}")
                self._print_details(signal.details)
            print()
        
        if info:
            print("ℹ️  INFORMATION")
            print("-" * 70)
            for signal in info:
                print(f"  [{signal.name.upper()}] {signal.message}")
            print()
        
        print("=" * 70)
        print("ACTIONABLE RECOMMENDATIONS")
        print("=" * 70)
        self._generate_recommendations(result['signals'])
        print()
    
    def _print_details(self, details: Dict[str, Any]):
        """Print relevant details for a signal.
        
        The nitty gritty stuff that might actually be useful.
        """
        if 'high_missing_columns' in details and details['high_missing_columns']:
            print(f"    → Columns with >50% missing: {details['high_missing_columns'][:3]}")
        
        if 'columns_with_outliers' in details and details['columns_with_outliers']:
            top_outliers = sorted(details['columns_with_outliers'].items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
            for col, pct in top_outliers:
                print(f"    → {col}: {pct:.1f}% outliers")
        
        if 'warnings' in details and details['warnings']:
            for warning in details['warnings'][:3]:
                print(f"    → {warning}")
        
        if 'class_distribution' in details:
            dist = details['class_distribution']
            if len(dist) <= 5:
                print(f"    → Class distribution: {dist}")
    
    def _generate_recommendations(self, signals: List[QualitySignal]):
        """Generate actionable next steps.
        
        Because "fix your data" isnt helpful enough.
        """
        recommendations = []
        
        for signal in signals:
            if signal.severity == 'critical':
                if signal.name == 'missing_values':
                    recommendations.append("1. Impute or remove columns/rows with >50% missingness")
                    recommendations.append("2. Investigate systematic patterns in missing data")
                elif signal.name == 'outliers':
                    recommendations.append("1. Review extreme outliers - are they errors or real?")
                    recommendations.append("2. Consider robust scaling or winsorization")
                elif signal.name == 'leakage':
                    recommendations.append("1. REMOVE features perfectly correlated with target")
                    recommendations.append("2. Verify temporal ordering of data splits")
                elif signal.name == 'label_noise':
                    recommendations.append("1. Balance classes via resampling or class weights")
                    recommendations.append("2. Collect more data for minority classes")
        
        if not recommendations:
            recommendations.append("✓ No critical actions required")
            recommendations.append("✓ Consider standard preprocessing: scaling, encoding")
        
        for rec in recommendations:
            print(f"  {rec}")


# Example usage
if __name__ == "__main__":
    # Create synthetic example dataset with quality issues
    # (because real data is too boring/confidential)
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'feature_1': np.random.normal(100, 15, n_samples),
        'feature_2': np.random.exponential(5, n_samples),
        'feature_3': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'leaky_feature': np.random.normal(50, 10, n_samples),  # Will correlate with target
        'id_column': range(n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # Imbalanced
    }
    
    df = pd.DataFrame(data)
    
    # Inject quality issues (for demo purposes)
    df.loc[np.random.choice(df.index, 100), 'feature_1'] = np.nan  # 20% missing
    df.loc[np.random.choice(df.index, 20), 'feature_2'] = 1000  # Outliers
    df['leaky_feature'] = df['target'] * 100 + np.random.normal(0, 1, n_samples)  # Leakage
    
    # Run assessment
    engine = DataQualityEngine()
    result = engine.assess(df, target_col='target')
    engine.print_report(result)