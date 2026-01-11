# CoT: Correlation Tuning for Fairness-Performance Trade-Off in ML Software

[![Paper](https://img.shields.io/badge/Paper-ICSE%202026-blue)](https://doi.org/xxxx)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Fairness Is Not Just Ethical: Performance Trade-Off via Data Correlation Tuning to Mitigate Bias in ML Software**
> 
> *Accepted at ICSE 2026*

This repository provides the replication package for our ICSE 2026 paper. CoT (Correlation Tuning) is a pre-processing bias mitigation approach that tunes the correlation between sensitive attributes and class labels in training data to achieve a better fairness-performance trade-off.

![CoT Overview](CoT_Overview.png)

---

## 📋 Table of Contents

- [Repository Structure](#-repository-structure)
- [Experimental Environment](#-experimental-environment)
- [Datasets](#-datasets)
- [Replication Guide](#-replication-guide)
  - [RQ1: Single-Attribute Fairness](#rq1-single-attribute-fairness)
  - [RQ2: Comparison with State-of-the-Art](#rq2-comparison-with-state-of-the-art)
  - [RQ3: Multi-Attribute Fairness](#rq3-multi-attribute-fairness)
  - [RQ4: Generalization to Deep Learning and LLMs](#rq4-generalization-to-deep-learning-and-llms)
  - [RQ5: Trade-Off Analysis](#rq5-trade-off-analysis)
- [Results](#-results)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## 📁 Repository Structure

```
ICSE26-CoT/
├── CoT_Overview.png                    # Overview figure of CoT approach
├── Preprint-CoT.pdf                    # Preprint of the paper
├── README.md                           # This file
└── CoT_Materia_New.zip                 # Replication materials (extract before use)
    │
    ├── 1_CoT_Code/                     # Source code
    │   ├── SingleAttr/                 # Single protected attribute experiments
    │   │   ├── CoT_Phi.py              # CoT with fixed α (Φ-coefficient based)
    │   │   ├── CoT_Opt.py              # CoT with optimized α (PSO-based)
    │   │   ├── CoT_Phi_dl.py           # CoT_Phi for deep learning models
    │   │   ├── CoT_Opt_dl.py           # CoT_Opt for deep learning models
    │   │   ├── CoT_Phi_Para.py         # Parameter search for CoT_Phi
    │   │   ├── CoT_Opt_Para.py         # Parameter search for CoT_Opt
    │   │   ├── Measure.py              # Fairness metrics computation
    │   │   ├── utility.py              # ML classifiers (LR, RF, SVM, XGB, LGBM)
    │   │   └── utility_dl.py           # Deep learning model architecture
    │   │
    │   └── MultiAttr/                  # Multiple protected attributes experiments
    │       ├── Cot_Phi.py              # CoT_Phi for multi-attribute protection
    │       ├── Cot_Opt.py              # CoT_Opt for multi-attribute protection
    │       ├── Cot_Phi_dl.py           # Multi-attr CoT_Phi for DL
    │       ├── Cot_Opt_dl.py           # Multi-attr CoT_Opt for DL
    │       ├── Measure_new.py          # Multi-attr fairness metrics
    │       └── ...
    │
    ├── 2_ResultForPaper/               # Aggregated results for paper tables/figures
    │   ├── RQ1_result.txt              # Results for RQ1 (Table 1)
    │   ├── RQ2_result                  # Results for RQ2 (Tables 4-5)
    │   ├── RQ3_result                  # Results for RQ3 (Figure 1)
    │   ├── RQ4_result1                 # Results for RQ4 (Tables 2-3)
    │   ├── RQ4_result2                 # Additional RQ4 results
    │   ├── RQ5_result1.txt             # Results for RQ5
    │   └── all_metrics_effect.txt      # Effect size analysis
    │
    ├── 3_RawResult/                    # Raw experimental results
    │   ├── SingleAttr/                 # Single-attribute results (320 files)
    │   ├── MultiAttr/                  # Multi-attribute results (120 files)
    │   └── LLM/                        # LLM experiment results (17 files)
    │
    ├── 4_MOO_Exploration/              # Multi-objective optimization exploration
    │   ├── CoT_Opt_Para.py             # Combined fairness metrics
    │   ├── CoT_Opt_Para_SPD.py         # SPD-focused optimization
    │   ├── CoT_Opt_Para_AOD.py         # AOD-focused optimization
    │   └── CoT_Opt_Para_EOD.py         # EOD-focused optimization
    │
    ├── 5_FinedTinyLlama/               # LLM fine-tuning experiments
    │   ├── Fine-tune-Script.ipynb      # Jupyter notebook for TinyLlama fine-tuning
    │   ├── CoT_Opt_Para.json           # Optimal α parameters
    │   └── Dataset_TempFile/           # Preprocessed data for LLM
    │
    └── 6_Dataset/                      # Preprocessed datasets
        ├── adult_processed.csv
        ├── compas_processed.csv
        ├── default_processed.csv
        ├── mep1_processed.csv
        ├── mep2_processed.csv
        └── z_*_{train,test}.csv        # Train/test splits for LLM experiments
```

---

## 🔧 Experimental Environment

### Requirements

- **Python**: 3.12 (recommended)
- **OS**: Linux/macOS/Windows

### Installation

1. **Create a virtual environment** (recommended):

```bash
conda create --name cot python=3.12
conda activate cot
```

2. **Install AIF360 and dependencies**:

```bash
pip install aif360
pip install scikit-learn
pip install numpy
pip install pandas
pip install shapely
pip install matplotlib
pip install scikit-opt        # For PSO optimization
pip install xgboost
pip install lightgbm
pip install protobuf==3.20.0
```

3. **For deep learning experiments** (optional):

```bash
pip install tensorflow>=2.0
# or
pip install keras
```

4. **For LLM experiments** (optional):

```bash
pip install transformers accelerate peft bitsandbytes
```

---

## 📊 Datasets

We evaluate CoT on **five** benchmark datasets commonly used in ML fairness research:

| Dataset | Task | Protected Attributes | # Samples | # Features |
|---------|------|---------------------|-----------|------------|
| **Adult** | Income prediction | Sex, Race | 48,842 | 14 |
| **COMPAS** | Recidivism prediction | Sex, Race | 7,214 | 28 |
| **Default** | Credit default prediction | Sex, Age | 30,000 | 24 |
| **MEPS-15** (mep1) | Healthcare utilization | Sex, Race | 15,830 | 138 |
| **MEPS-16** (mep2) | Healthcare utilization | Sex, Race | 15,675 | 138 |

All datasets are preprocessed and included in `6_Dataset/`. The preprocessing scripts handle:
- Binary encoding of protected attributes
- Label standardization to `Probability` column
- Missing value handling

---

## 🚀 Replication Guide

First, extract the replication materials:

```bash
unzip "CoT_Materia_New.zip"
cd CoT_Materia_New/1_CoT_Code/SingleAttr
```

### RQ1: Single-Attribute Fairness

**Goal**: Evaluate CoT's effectiveness in mitigating bias for a single protected attribute.

**Scripts**: 
- `CoT_Phi.py` - Uses Φ-coefficient to determine the tuning parameter α
- `CoT_Opt.py` - Uses PSO to optimize α

**Arguments**:
| Argument | Description | Options |
|----------|-------------|---------|
| `-d` | Dataset | `adult`, `compas`, `default`, `mep1`, `mep2` |
| `-c` | ML Classifier | `lr`, `rf`, `svm`, `xgb`, `lgbm` |
| `-p` | Protected Attribute | Dataset-dependent (e.g., `sex`, `race`, `age`) |

**Example Commands**:

```bash
# CoT_Phi with different classifiers
python CoT_Phi.py -d adult -c lr -p sex
python CoT_Phi.py -d adult -c rf -p race
python CoT_Phi.py -d compas -c svm -p sex
python CoT_Phi.py -d default -c lr -p age
python CoT_Phi.py -d mep1 -c rf -p sex
python CoT_Phi.py -d mep2 -c svm -p race

# CoT_Opt (PSO-optimized)
python CoT_Opt.py -d adult -c lr -p sex
python CoT_Opt.py -d compas -c rf -p race
```

**Full experiment** (all combinations):

```bash
for dataset in adult compas default mep1 mep2; do
    for clf in lr rf svm; do
        for attr in sex race age; do
            python CoT_Phi.py -d $dataset -c $clf -p $attr 2>/dev/null || true
            python CoT_Opt.py -d $dataset -c $clf -p $attr 2>/dev/null || true
        done
    done
done
```

### RQ2: Comparison with State-of-the-Art

We compare CoT with six state-of-the-art bias mitigation methods:
- **REW** (Reweighing)
- **ROC** (Reject Option Classification)
- **ADV** (Adversarial Debiasing)
- **Fairway**
- **Fair-SMOTE**
- **MAAT**

The baseline implementations are available at: [MAAT Repository](https://github.com/chenzhenpeng18/FSE22-MAAT)

### RQ3: Multi-Attribute Fairness

**Goal**: Evaluate CoT's ability to simultaneously protect multiple sensitive attributes.

**Scripts** (in `MultiAttr/` folder):

```bash
cd ../MultiAttr

# Protect both sex and race simultaneously
python Cot_Phi.py -d adult -c lr
python Cot_Phi.py -d adult -c rf
python Cot_Phi.py -d compas -c lr
python Cot_Phi.py -d compas -c svm
python Cot_Phi.py -d mep1 -c rf
python Cot_Phi.py -d mep2 -c rf
```

**Note**: Multi-attribute scripts automatically handle both protected attributes defined for each dataset.

### RQ4: Generalization to Deep Learning and LLMs

#### Deep Learning Models

```bash
cd SingleAttr

# Deep learning with CoT_Phi
python CoT_Phi_dl.py -d adult -c dl -p sex
python CoT_Phi_dl.py -d adult -c dl -p race
python CoT_Phi_dl.py -d compas -c dl -p sex
python CoT_Phi_dl.py -d compas -c dl -p race
python CoT_Phi_dl.py -d default -c dl -p sex
python CoT_Phi_dl.py -d default -c dl -p age
python CoT_Phi_dl.py -d mep1 -c dl -p sex
python CoT_Phi_dl.py -d mep1 -c dl -p race
python CoT_Phi_dl.py -d mep2 -c dl -p sex
python CoT_Phi_dl.py -d mep2 -c dl -p race
```

#### Large Language Models (TinyLlama)

The LLM experiments use TinyLlama-1.1B with LoRA fine-tuning. Run in Google Colab with GPU:

```bash
cd 5_FinedTinyLlama
# Open Fine-tune-Script.ipynb in Jupyter/Colab
```

The notebook:
1. Fine-tunes TinyLlama on original and CoT-processed training data
2. Evaluates fairness metrics on test sets
3. Compares performance between original and CoT-enhanced models

### RQ5: Trade-Off Analysis

Analyze the fairness-performance trade-off with different weight configurations:

```bash
cd 4_MOO_Exploration

# Different optimization objectives
python CoT_Opt_Para.py      # Combined metrics
python CoT_Opt_Para_SPD.py  # SPD-focused
python CoT_Opt_Para_AOD.py  # AOD-focused
python CoT_Opt_Para_EOD.py  # EOD-focused
```

---

## 📈 Results

### Output Format

Each experiment generates a `.txt` file with the following metrics:
- **Performance**: Accuracy, Recall, Precision, F1-score, MCC
- **Fairness (Primary Attribute)**: SPD, AOD, EOD
- **Fairness (Secondary Attribute)**: SPD2, AOD2, EOD2
- **Subgroup Metrics**: TPR/TNR for privileged/unprivileged groups

### Key Findings

| Method | Fairness Improvement | Performance Retention |
|--------|---------------------|----------------------|
| **CoT_Opt** | 97.8% tasks show decrease | 83.3% with large effect |
| **CoT_Phi** | 92.2% tasks show decrease | 81.7% with large effect |
| MirrorFair | 92.2% | 77.8% |
| FairMask | 90.6% | 67.8% |

---

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{cot2026icse,
  title={Fairness Is Not Just Ethical: Performance Trade-Off via Data Correlation Tuning to Mitigate Bias in ML Software},
  author={[Authors]},
  booktitle={Proceedings of the 48th International Conference on Software Engineering (ICSE)},
  year={2026},
  organization={IEEE/ACM}
}
```

---

## 🙏 Acknowledgments

We thank the authors of the following works for their open-source contributions:

- **[AIF360](https://github.com/Trusted-AI/AIF360)** - IBM AI Fairness 360 toolkit
- **[Fairway](https://doi.org/10.1145/3368089.3409697)** - Chakraborty et al., ASE 2020
- **[Fair-SMOTE](https://doi.org/10.1145/3468264.3468537)** - Chakraborty et al., FSE 2021
- **[Fairea](https://doi.org/10.1145/3468264.3468565)** - Hort et al., FSE 2021
- **[MAAT](https://github.com/chenzhenpeng18/FSE22-MAAT)** - Chen et al., FSE 2022
- **[MirrorFair](https://doi.org/xxxx)** - Recent fairness method
- **[FairMask](https://doi.org/xxxx)** - Peng et al.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

**License**: MIT License
