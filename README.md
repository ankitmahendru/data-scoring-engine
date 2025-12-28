# 📊 Data Quality Scoring Engine

> *Because apparently we need to check data BEFORE feeding it to models. Who knew? (Everyone. Everyone knew.)*

<div align="center">

</div>

---

## 🧐 What is this?

The **Data Quality Scoring Engine** is a pragmatic system designed to stop you from wasting GPU hours on garbage data. Instead of flying blind, this engine generates a "Readiness Score" (0-100) by analyzing datasets for common ML deal-breakers: data sparsity, extreme anomalies, feature leakage, and label noise.

It’s built for the cynical data scientist who knows that hand-annotated labels are usually buggy, and that "99.9% accuracy" is just code for "I have leakage".

---

## ✨ Key Features

* **Readiness Scoring**: An aggregate 0-100 score that categorizes datasets from "Production-grade" to "Critical issues must be resolved".
* **Leakage Detection**: Identifies "too good to be true" signals like perfect target correlations or suspicious ID-like columns.
* **Outlier Analysis**: Uses the classic IQR approach for univariate detection and Isolation Forests for the multivariate "fancy stuff".
* **Universal Runner**: A "Swiss Army knife" script that can pull any dataset directly from Kaggle and run an interactive quality assessment.
* **Actionable Recommendations**: Doesn't just tell you your data is bad; it provides specific next steps like resampling, winsorization, or feature removal.

---

## 📂 Project Structure

Behold, the blueprint of a system that actually checks its inputs:

```bash
data-scoring-engine/
├── dq_results/             # Where your data's dirty secrets are stored
├── data_quality_engine.py  # The heart of the system (Analyzers & Scoring)
├── universal_dq_runner.py  # Kaggle-compatible CLI for ANY dataset
├── run_kaggle_dataset.py   # Specialized runner for the Housing dataset
├── House_Rent_Dataset.csv  # Sample local dataset for testing
└── README.md               # This masterpiece.

```

---

## 🛠 Tech Stack

| Layer | Technology | Why? |
| --- | --- | --- |
| **Data Engine** | Pandas | For wrangling dataframes like a digital cowboy. |
| **Intelligence** | Scikit-Learn | Specifically `IsolationForest` for spotting weirdness. |
| **CLI / API** | Kaggle API | Because manual downloads are so last century. |
| **Logic** | Python 3.8+ | Typings and Dataclasses for a bit of professional flair. |

---

## 💿 Installation & Setup

Ready to face the truth about your data?

### 1. Clone & Prep

```bash
git clone https://github.com/ankitmahendru/data-scoring-engine.git
cd data-scoring-engine
pip install pandas numpy scikit-learn kaggle

```

### 2. Run a Local Test

```bash
python data_quality_engine.py

```

### 3. The Universal Kaggle Method

```bash
# Analyze any Kaggle dataset with auto-detection
python universal_dq_runner.py -d pranavshinde36/india-house-rent-prediction --auto-detect

```

---

## 🎮 How it Evaluates

The engine runs a battery of tests to determine your fate:

1. **Missing Value Analyzer**: Detects rows/columns that are basically empty.
2. **Outlier Analyzer**: Flags anomalous density and distribution issues.
3. **Leakage Detector**: The "why is my model TOO good" check.
4. **Label Noise Estimator**: Checks for severe class imbalance and label consistency.

---

## 🤝 Contribution Guide

Think my math is off? Or maybe the Isolation Forest needs more trees?

1. **Fork it.**
2. **Branch it** (`git checkout -b feature/better-scoring`).
3. **Commit it** (Keep it clean, or I'll reject it).
4. **Push it.**
5. **PR it.**

---

<div align="center">

**Made with love (and a deep skepticism of all data) by PadhoAI** ❤️

</div>
