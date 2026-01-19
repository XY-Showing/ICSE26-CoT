# Fairness Is Not Just Ethical: Performance Trade-Off via Data Correlation Tuning to Mitigate Bias in ML Software

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2512.21348)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow)](http://creativecommons.org/licenses/by/4.0/)

> **Fairness Is Not Just Ethical: Performance Trade-Off via Data Correlation Tuning to Mitigate Bias in ML Software**
> 
> *Ying Xiao, Shangwen Wang, Sicen Liu, Dingyuan Xue, Xian Zhan, Yepang Liu, Jie M. Zhang*
>
> *ICSE 2026 (April 12-18, 2026, Rio de Janeiro, Brazil)*

This repository provides the replication package for our ICSE 2026 paper. **CoT (Correlation Tuning)** is a pre-processing bias mitigation approach that tunes the correlation between sensitive attributes and class labels in training data to achieve a better fairness-performance trade-off.

![CoT Overview](CoT_Overview.png)

---

## 📋 Table of Contents

- [Repository Structure](#-repository-structure)
- [Experimental Environment](#-experimental-environment)
- [Datasets](#-datasets)
- [Research Questions & Replication](#-research-questions--replication)
  - [RQ1: Impact on Performance and Fairness](#rq1-impact-on-performance-and-fairness)
  - [RQ2: Bias Mitigation Effectiveness](#rq2-bias-mitigation-effectiveness)
  - [RQ3: Performance-Fairness Trade-Off](#rq3-performance-fairness-trade-off)
  - [RQ4: Multiple Sensitive Attributes](#rq4-multiple-sensitive-attributes)
  - [RQ5: Robustness Analysis](#rq5-robustness-analysis)
- [Results](#-results)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## 📁 Repository Structure

```
ICSE26-CoT/
├── CoT_Overview.png                    # Overview figure of CoT approach
├── Preprint-CoT.pdf                    # Paper preprint
├── README.md                           # This file
└── CoT_Materia_New.zip                 # Replication materials (extract before use)
    │
    ├── 1_CoT_Code/                     # Source code
    │   ├── SingleAttr/                 # Single protected attribute experiments
    │   │   ├── CoT_Phi.py              # CoT with Φ-coefficient based α
    │   │   ├── CoT_Opt.py              # CoT with multi-objective optimization α
    │   │   ├── CoT_Phi_dl.py           # CoT_Phi for deep learning models
    │   │   ├── CoT_Opt_dl.py           # CoT_Opt for deep learning models
    │   │   ├── CoT_Phi_Para.py         # Parameter search for CoT_Phi
    │   │   ├── CoT_Opt_Para.py         # Parameter search for CoT_Opt
    │   │   ├── Measure.py              # Fairness metrics computation
    │   │   ├── utility.py              # ML classifiers (LR, RF, SVM, XGB, LGBM)
    │   │   └── utility_dl.py           # Deep learning model architecture
    │   │
    │   └── MultiAttr/                  # Multiple protected attributes experiments (RQ4)
    │       ├── Cot_Phi.py              # CoT_Phi for multi-attribute protection
    │       ├── Cot_Opt.py              # CoT_Opt for multi-attribute protection
    │       ├── Measure_new.py          # Multi-attr fairness metrics
    │       └── ...
    │
    ├── 2_ResultForPaper/               # Aggregated results for paper tables/figures
    │   ├── RQ1_result.txt              # Results for RQ1 (Table 4)
    │   ├── RQ2_result                  # Results for RQ2 (Tables 5-6)
    │   ├── RQ3_result                  # Results for RQ3 (Figure 2)
    │   ├── RQ4_result1                 # Results for RQ4 (Tables 7-8)
    │   ├── RQ4_result2                 # RQ4 intersectional bias results
    │   ├── RQ5_result*.txt             # Results for RQ5 (Tables 9-10)
    │   └── all_metrics_effect.txt      # Effect size analysis
    │
    ├── 3_RawResult/                    # Raw experimental results
    │   ├── SingleAttr/                 # Single-attribute results (320 files)
    │   ├── MultiAttr/                  # Multi-attribute results (120 files)
    │   └── LLM/                        # LLM experiment results (17 files)
    │
    ├── 4_MOO_Exploration/              # Multi-objective optimization exploration
    │   ├── CoT_Opt_Para.py             # Combined fairness metrics optimization
    │   ├── CoT_Opt_Para_SPD.py         # SPD-focused optimization
    │   ├── CoT_Opt_Para_AOD.py         # AOD-focused optimization
    │   └── CoT_Opt_Para_EOD.py         # EOD-focused optimization
    │
    ├── 5_FinedTinyLlama/               # LLM fine-tuning experiments
    │   ├── Fine-tune-Script.ipynb      # Jupyter notebook for TinyLlama fine-tuning
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
pip install tensorflow keras
```

4. **For LLM experiments** (optional):

```bash
pip install transformers accelerate peft bitsandbytes
```

---

## 📊 Datasets

We evaluate CoT on **five** benchmark datasets commonly used in ML fairness research:

| Dataset | Task | Protected Attributes | # Samples | 
|---------|------|---------------------|-----------|
| **Adult** | Income prediction | Sex, Race | 48,842 | 
| **COMPAS** | Recidivism prediction | Sex, Race | 7,214 | 
| **Default** | Credit default prediction | Sex, Age | 30,000 | 
| **MEPS-15** (mep1) | Healthcare utilization | Sex, Race | 15,830 | 
| **MEPS-16** (mep2) | Healthcare utilization | Sex, Race | 15,675 | 

All datasets are preprocessed and included in `6_Dataset/`. 

---

## 🔬 Research Questions & Replication

First, extract the replication materials:

```bash
unzip "CoT_Materia_New.zip"
cd CoT_Materia_New/1_CoT_Code/SingleAttr
```

### RQ1: Impact on Performance and Fairness

> **RQ1**: *What is the impact of CoT on model performance and fairness?*

This RQ investigates the quantitative impact of CoT on model performance and fairness by comparing various metrics between CoT-enhanced models and original models without bias mitigation.

**Scripts**: `CoT_Phi.py`, `CoT_Opt.py`

| Argument | Description | Options |
|----------|-------------|---------|
| `-d` | Dataset | `adult`, `compas`, `default`, `mep1`, `mep2` |
| `-c` | ML Classifier | `lr`, `rf`, `svm`, `xgb`, `lgbm` |
| `-p` | Protected Attribute | `sex`, `race`, `age` |

**Example Commands**:

```bash
# CoT-Phi (Φ-coefficient based)
python CoT_Phi.py -d adult -c lr -p sex
python CoT_Phi.py -d adult -c rf -p race
python CoT_Phi.py -d compas -c svm -p sex

# CoT-Opt (multi-objective optimization)
python CoT_Opt.py -d adult -c lr -p sex
python CoT_Opt.py -d compas -c rf -p race
```

**Results**: See `2_ResultForPaper/RQ1_result.txt` (corresponds to **Table 4** in paper)

---

### RQ2: Bias Mitigation Effectiveness

> **RQ2**: *What is the bias mitigation effectiveness of CoT compared with existing methods?*

This RQ examines CoT's effectiveness in mitigating bias by analyzing the proportion of scenarios with fairness improvements and the degree of changes in SPD, AOD, and EOD fairness metrics.

**Baseline Methods**:
- Fair-SMOTE, FairGenerate, LTDD
- FairMask, MirrorFair


**Results**: See `2_ResultForPaper/RQ2_result` (corresponds to **Tables 5-6** in paper)

---

### RQ3: Performance-Fairness Trade-Off

> **RQ3**: *What is the performance-fairness trade-off of CoT compared with existing methods?*

This RQ investigates the advantages of CoT in balancing model performance and fairness using the **Fairea Trade-off** measurement tool, which classifies outcomes into five categories: *win-win*, *good*, *bad*, *inverted*, and *lose-lose*.

**Results**: See `2_ResultForPaper/RQ3_result` (corresponds to **Figure 2** in paper)

---

### RQ4: Multiple Sensitive Attributes

> **RQ4**: *What is the effectiveness of CoT in handling multiple sensitive attributes?*

This RQ examines:
1. The potential negative side effects on **unconsidered** sensitive attributes
2. The effectiveness in mitigating **intersectional bias**

**Scripts** (in `MultiAttr/` folder):

```bash
cd ../MultiAttr

# Multi-attribute protection
python Cot_Phi.py -d adult -c lr
python Cot_Phi.py -d adult -c rf
python Cot_Phi.py -d compas -c lr
python Cot_Phi.py -d compas -c svm
```

**Note**: Multi-attribute scripts automatically protect both sensitive attributes defined for each dataset.

**Results**: See `2_ResultForPaper/RQ4_result1` and `RQ4_result2` (corresponds to **Tables 7-8** in paper)

---

### RQ5: Robustness Analysis

> **RQ5**: *What is the robustness of CoT under overfitting risks and realistic data conditions?*

This RQ investigates the robustness of CoT by evaluating whether its effectiveness remains consistent under:
- **Overfitting conditions**: Using modern classifiers like XGBoost
- **Realistic data conditions**: With noise and distribution shifts

**Deep Learning Experiments**:

```bash
python CoT_Phi_dl.py -d adult -c dl -p sex
python CoT_Phi_dl.py -d compas -c dl -p race
python CoT_Opt_dl.py -d default -c dl -p age
```

**LLM Experiments** (TinyLlama fine-tuning):

```bash
cd 5_FinedTinyLlama
# Run Fine-tune-Script.ipynb in Jupyter/Colab with GPU
```

**Results**: See `2_ResultForPaper/RQ5_result*.txt` (corresponds to **Tables 9-10** in paper)

---

## 📈 Results

### Output Format

Each experiment generates a `.txt` file with the following metrics:

| Category | Metrics |
|----------|---------|
| **Performance** | Accuracy, Recall, Precision, F1-score, MCC |
| **Fairness (Primary)** | SPD, AOD, EOD |
| **Fairness (Secondary)** | SPD2, AOD2, EOD2 |
| **Subgroup** | TPR_P, TPR_U, TNR_P, TNR_U, Acc_P, Acc_U |

### Key Findings

| Metric | CoT-Opt | CoT-Phi | Best Baseline |
|--------|---------|---------|---------------|
| Fairness Improvement Rate | **97.8%** | 92.2% | 92.2% (MirrorFair) |
| Large Effect Size | **83.3%** | 81.7% | 77.8% (MirrorFair) |
| SPD Reduction | -49% | -54% | -46% (MirrorFair) |
| AOD Reduction | **-58%** | -55% | -52% (MirrorFair) |
| EOD Reduction | **-51%** | -41% | -50% (MirrorFair) |

---

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{xiao2026cot,
  title={Fairness Is Not Just Ethical: Performance Trade-Off via Data Correlation Tuning to Mitigate Bias in ML Software},
  author={Xiao, Ying and Wang, Shangwen and Liu, Sicen and Xue, Dingyuan and Zhan, Xian and Liu, Yepang and Zhang, Jie M.},
  booktitle={Proceedings of the 48th IEEE/ACM International Conference on Software Engineering (ICSE)},
  year={2026},
  organization={IEEE/ACM}
}
```

**arXiv**: https://arxiv.org/abs/2512.21348

---

## 🙏 Acknowledgments

We thank the authors of the following works for their open-source contributions:

- **[AIF360](https://github.com/Trusted-AI/AIF360)** - IBM AI Fairness 360 toolkit
- **[Fair-SMOTE](https://doi.org/10.1145/3468264.3468537)** - Chakraborty et al., FSE 2021
- **[FairGenerate](https://dl.acm.org/doi/10.1145/3730579)** - Joshi & Kumar, TOSEM 2025
- **[LTDD](https://dl.acm.org/doi/10.1145/3510003.3510091)** - Li et al., ICSE 2022
- **[Fairea](https://doi.org/10.1145/3468264.3468565)** - Hort et al., FSE 2021
- **[FairMask](https://ieeexplore.ieee.org/document/9951398)** - Peng et al., TSE 2022
- **[MirrorFair](https://dl.acm.org/doi/10.1145/3660801)** - Xiao et al., FSE 2024

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors at: ying.1.xiao@kcl.ac.uk

---

**License**: [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)
