# Data Quality Scoring Engine

A pragmatic, ML-adjacent system for assessing dataset readiness through quality signals rather than predictions.

Because apparently checking your data *before* training models is too radical of an idea.

## Philosophy

This is **not** an ML model. It's a deterministic heuristic system that catches data quality issues before they become expensive modeling failures. It prioritizes actionable warnings over statistical purity.

Think of it as a smoke detector for your datasets. Except it actually works.

## Installation

```bash
pip install pandas numpy scikit-learn
```

If you're on some weird python version and this breaks, that's between you and your environment manager.

## Quick Start

```python
from data_quality_engine import DataQualityEngine
import pandas as pd

# Load your dataset
df = pd.read_csv('your_data.csv')

# Run assessment
engine = DataQualityEngine()
result = engine.assess(df, target_col='your_target_column')

# Print human-readable report
engine.print_report(result)

# Access structured results (for the nerds)
print(f"Score: {result['score']}/100")
print(f"Readiness: {result['readiness']}")
```

## Core Quality Signals

### 1. Missing Value Patterns (30% weight)
**What it detects:**
- Overall data sparsity (the holes in your data)
- Columns with systematic missingness (>50%)
- Rows with excessive missing values (>70%)

**Assumptions:**
- Missing data is often not random (MNAR/MAR) - it has patterns
- Columns with >50% missing are rarely salvageable (just delete them)
- Sparse data degrades model performance non-linearly (aka exponentially worse)

**False Positives:**
- Legitimately sparse features (e.g., optional medical tests)
- Intentionally missing values in survey data (like "prefer not to answer")
- Features where missingness is informative (missingness itself is a signal)

**Mitigation:** Review columns flagged as "high_missing" manually. Use your brain.

---

### 2. Outlier Density (25% weight)
**What it detects:**
- Univariate outliers using IQR method (3×IQR threshold)
- Multivariate anomalies via Isolation Forest (fancy tree stuff)
- Per-column contamination rates

**Assumptions:**
- Outliers beyond 3×IQR are likely errors or rare events
- >15% outliers suggests data quality issues, not interesting edge cases
- Multivariate outliers may indicate recording errors (or someone mashed the keyboard)

**False Positives:**
- Heavy-tailed distributions (e.g., income, web traffic) - these are naturally weird
- Legitimate rare events in fraud/medical data (the whole point of those domains)
- Small sample sizes (<100 rows) - everything looks like an outlier

**Mitigation:** 
- Review flagged columns with domain knowledge (ask someone who knows the data)
- Consider log-transformations for skewed features
- Adjust IQR multiplier for heavy-tailed domains (change the 3 to something else)

---

### 3. Feature Leakage Indicators (30% weight)
**What it detects:**
- Perfect correlations with target (>0.99) - the smoking gun
- Suspiciously high correlations (>0.95) - probably also bad
- Duplicate columns (why do people do this?)
- ID-like columns (>95% unique values with 'id', 'key' patterns in name)

**Assumptions:**
- Perfect correlation = future information leaked into features
- ID columns should never be predictive (if they are, something's wrong)
- Duplicate columns indicate preprocessing errors (or copy-paste gone wrong)

**False Positives:**
- Derived features intentionally correlated with target (like "total_price" from "price_per_unit" × "units")
- Legitimate high correlations (e.g., temperature in Celsius vs Fahrenheit - duh)
- Composite keys that aren't true IDs (like "state_county" codes)

**Mitigation:**
- Verify temporal causality for high-correlation features (did you peek at the future?)
- Check if "ID" columns are actually categorical features (sometimes "customer_id" is legit)
- Validate feature engineering pipeline (trace back where everything came from)

**CRITICAL:** This check requires specifying `target_col`. Without it, leakage detection is skipped (and you're flying blind).

---

### 4. Label Noise Estimation (15% weight)
**What it detects:**
- Class imbalance ratios (how skewed are your classes)
- Target distribution statistics
- Coefficient of variation for regression targets

**Assumptions:**
- Imbalance >100:1 requires specialized handling (good luck)
- Imbalance >10:1 likely hurts model performance (your model will just predict majority class)
- Extreme imbalance often indicates annotation errors (or you scraped the data wrong)

**False Positives:**
- Naturally rare events (fraud, disease diagnosis) - imbalance is expected
- Intentionally filtered datasets (like "only show me the fraudulent transactions")
- Early-stage data collection (you just started and don't have much data yet)

**Mitigation:**
- Confirm imbalance reflects real-world distribution (is this actually how the world works?)
- Plan for stratified sampling or SMOTE (fancy resampling techniques)
- Consider adjusting classification thresholds (you dont always need 0.5)

---

## Scoring Logic

### Component Scores
Each signal produces a 0-100 score:
- **90-100**: Excellent, minimal issues (you won the data lottery)
- **70-89**: Good with minor concerns (ship it)
- **50-69**: Problematic, needs attention (fix before demo day)
- **0-49**: Critical, must fix before modeling (don't even try)

### Aggregate Score
Weighted average of component scores:
```
Final Score = 0.30×Missing + 0.25×Outliers + 0.30×Leakage + 0.15×Labels
```

Weights reflect typical impact on modeling outcomes:
- Leakage and missing values cause catastrophic failures → high weight (these will ruin your day)
- Outliers and label noise are more recoverable → lower weight (annoying but fixable)

### Readiness Categories
- **85-100**: READY - Production-grade dataset (send it)
- **70-84**: CAUTION - Addressable issues present (proceed with eyes open)
- **50-69**: NEEDS WORK - Significant quality gaps (roll up your sleeves)
- **0-49**: NOT READY - Critical issues block modeling (back to the drawing board)

---

## Limitations & Known Issues

### What This System **Does Not** Do
1. **No causality detection**: Cannot tell if high correlation is leakage or legitimate (you need domain knowledge)
2. **No domain validation**: Doesn't know if your "outliers" are actually errors (again, use your brain)
3. **No data drift**: Only analyzes single snapshots, not temporal changes (run it multiple times yourself)
4. **No feature engineering**: Won't suggest new features or transformations (that's your job)
5. **No guarantee**: A score of 100 doesn't guarantee good model performance (but it helps)

### Edge Cases
- **Small datasets** (<100 rows): Outlier detection becomes unreliable (not enough data to be statistical)
- **Wide datasets** (>1000 columns): Correlation checks may be slow (grab coffee)
- **Text features**: Ignored entirely; only numeric/categorical analyzed (NLP is hard)
- **Time series**: No special handling for autocorrelation or seasonality (just treats it like regular data)
- **Multi-target**: Only supports single target column (pick one)

### Computational Complexity
- **Missing values**: O(n×m) where n=rows, m=columns
- **Outliers**: O(n×m) + O(n×log(n)) for Isolation Forest
- **Leakage**: O(m²) for correlation matrix (slow for wide data)
- **Labels**: O(n) for distribution analysis

Expect ~1-5 seconds for typical datasets (10K rows × 50 cols). If its slower, your data is huge or your laptop is potato.

---

## Interpreting Results

### Critical Issues (🚨)
**Drop everything and fix these.** Your model will fail or leak. No exceptions.

Example: "Perfect correlation with target: ['total_purchase_amount']"
→ **Action**: This feature contains the answer. Remove it or verify it's not future data.

### Warnings (⚠️)
**Address before production.** May degrade performance or cause subtle bugs. Probably wont explode but still bad.

Example: "Moderate outliers: 12.3% anomalies detected"
→ **Action**: Investigate whether outliers are errors. Consider winsorization (capping extreme values).

### Info (ℹ️)
**For awareness only.** No immediate action required. You're probably fine.

Example: "Acceptable missingness: 3.2% missing values"
→ **Action**: Proceed with standard imputation strategies (mean/median/mode).

---

## Example Run Explained

```python
# Synthetic problematic dataset (intentionally broken for demo)
data = {
    'feature_1': [100, 102, np.nan, 98, ...],  # 20% missing
    'feature_2': [5, 6, 5, 1000, ...],          # Outliers injected
    'leaky_feature': [50, 150, 51, 149, ...],   # Perfectly correlates with target
    'id_column': [0, 1, 2, 3, ...],             # ID mistakenly included
    'target': [0, 1, 0, 1, ...]                 # 95% class 0, 5% class 1
}

result = engine.assess(df, target_col='target')
```

**Expected Output:**
```
OVERALL SCORE: 42.5/100
READINESS: NOT READY - Critical issues must be resolved

🚨 CRITICAL ISSUES
  [LEAKAGE] Strong leakage indicators detected
    → Perfect correlation with target: ['leaky_feature']
    → Potential ID column as feature: 'id_column'
  
  [LABEL_NOISE] Severe class imbalance: 19:1 ratio
    → Class distribution: {0: 475, 1: 25}

⚠️ WARNINGS
  [MISSING_VALUES] Moderate missingness: 20% missing values
    → Columns with >50% missing: ['feature_1']

ACTIONABLE RECOMMENDATIONS
  1. REMOVE features perfectly correlated with target
  2. Verify temporal ordering of data splits
  3. Balance classes via resampling or class weights
```

**Why this score?**
- Missing: 60/100 (20% missing is moderate)
- Outliers: 85/100 (few outliers, not terrible)
- Leakage: 20/100 (CRITICAL - perfect correlation, game over)
- Labels: 40/100 (CRITICAL - 19:1 imbalance, oof)

Weighted: 0.30×60 + 0.25×85 + 0.30×20 + 0.15×40 = **42.5**

---

## Extending the Engine

### Adding Custom Analyzers

```python
from data_quality_engine import QualitySignal

class CustomAnalyzer:
    def analyze(self, df: pd.DataFrame) -> QualitySignal:
        # Your custom logic here
        score = 85
        severity = 'warning'
        message = "Custom check result"
        details = {'key': 'value'}
        return QualitySignal('custom', score, severity, message, details)

# Register it (easy peasy)
engine.analyzers['custom'] = CustomAnalyzer()
```

### Adjusting Weights

```python
# In DataQualityEngine.assess() method, around line 260
weights = {
    'missing_values': 0.40,    # Increase if your domain has lots of missing data
    'outliers': 0.15,          # Decrease if outliers are expected (like finance)
    'leakage': 0.35,           # Keep high - leakage is always bad
    'label_noise': 0.10        # Adjust based on imbalance tolerance
}
```

---

## When to Use This Tool

### ✅ Good Use Cases
- Pre-modeling sanity checks (before you waste GPU time)
- Data intake validation pipelines (check before you commit)
- Debugging unexpected model performance (why is my model so bad/good?)
- Onboarding new data sources (trust but verify)
- Auditing third-party datasets (they always lie about quality)

### ❌ Poor Use Cases
- Replacing domain expertise (you still need to look at the data!)
- Real-time inference monitoring (use drift detection instead)
- Feature selection (use model-based methods like SHAP)
- Automated data cleaning (requires human judgement, sorry)

---

## FAQ

**Q: Why such harsh penalties for imbalance?**  
A: Because 19:1 imbalance means a dumb "always predict majority" baseline gets 95% accuracy. Your fancy model needs to beat that. Good luck.

**Q: My dataset scores 100 but my model still sucks. Why?**  
A: This checks data **quality**, not data **relevance**. You can have pristine data that's completely uninformative for your task. Like trying to predict stock prices from weather data - both might be perfect quality but totally unrelated.

**Q: Can I use this in production pipelines?**  
A: Yes, but treat it as a circuit breaker, not gospel. Route low-scoring datasets to manual review. Don't auto-reject without human eyes.

**Q: What if I disagree with a "critical" flag?**  
A: Override it! These are heuristics, not laws of physics. Document your reasoning and move on. You know your data better than this script does.

**Q: Does this replace EDA?**  
A: No. This catches **common** issues. You still need to plot distributions, check correlations, and think. Automation doesn't replace understanding.

**Q: Why is my score so low even though my data looks fine?**  
A: Check the detailed report. Maybe you have subtle issues you didn't notice. Or maybe the thresholds are too strict for your domain - feel free to adjust them.

**Q: Can I contribute?**  
A: Sure, but only if your PR actually adds value. No "refactoring" for style points.

---

## Contributing

This is a pragmatic tool, not an academic excercise. Useful additions:
- Better heuristics for leakage detection (e.g., temporal checks)
- Domain-specific analyzers (time series, NLP, images)
- Performance optimizations for very wide datasets
- More granular severity levels

Not interested in:
- Flashy visualizations (defeats the purpose)
- ML models to detect quality (too meta, too slow)
- "AI-powered" anything (buzzword bingo)

---

## License

MIT - Use it, break it, fix it, share it. Just don't blame me when it doesnt solve all your problems.

---

## Citation

If this saves you from a leaky model in production:
```
@software{data_quality_engine,
  title={Data Quality Scoring Engine: A Pragmatic Approach},
  year={2025},
  note={Because prevention is cheaper than debugging},
  author={Someone tired of bad data}
}
```

---

## Final Thoughts

Data quality is boring until it isn't. This tool won't solve all your problems, but it'll catch the obvious ones before they bite you. Use it as a first line of defense, not a replacement for critical thinking.

And remember: garbage in, garbage out. No amount of fancy models can fix fundamentally broken data.

Good luck out there. You're gonna need it.