# ICSE26-Fairness Is Not Just Ethical: Performance Trade-Off via Data Correlation Tuning to Mitigate Bias in ML Software
This repository releases the data, scripts, and results of the ICSE 2026 paper entitled “Fairness Is Not Just Ethical: Performance Trade-Off via Data Correlation Tuning to Mitigate Bias in ML Software” to facilitate replication of the experimental results and support further research on machine learning fairness.


# Replication instructions

This document contains information on how to install packages, datasets and replicate CoT.

## Experimental environment

We use Python 3.7 for our experiments. We use the IBM AI Fairness 360 (AIF360) toolkit for implementing bias mitigation methods and computing fairness metrics. 

Installation instructions for Python 3.7 and AIF360 can be found on https://github.com/Trusted-AI/AIF360. That page provides several ways for the installation. We recommend creating a virtual environment for it (as shown below), because AIF360 requires specific versions of many Python packages which may conflict with other projects on your system. If you would like to try other installation ways or encounter any errors during the installation process, please refer to the page (https://github.com/Trusted-AI/AIF360) for help.

#### Conda

Conda is recommended for all configurations. [Miniconda](https://conda.io/miniconda.html)
is sufficient (see [the difference between Anaconda and
Miniconda](https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda)
if you are curious) if you do not already have conda installed.

Then, to create a new Python 3.7 environment, run:

```bash
conda create --name aif360 python=3.7
conda activate aif360
```

The shell should now look like `(aif360) $`. To deactivate the environment, run:

```bash
(aif360)$ conda deactivate
```

The prompt will return to `$ `.

Note: Older versions of conda may use `source activate aif360` and `source
deactivate` (`activate aif360` and `deactivate` on Windows).

### Install with `pip`

To install the latest stable version from PyPI, run:

```bash
pip install aif360
```

[comment]: <> (This toolkit can be installed as follows:)

[comment]: <> (```)

[comment]: <> (pip install aif360)

[comment]: <> (```)

[comment]: <> (More information on installing AIF360 can be found on https://github.com/Trusted-AI/AIF360.)

In addition, we require the following Python packages. Note that TensorFlow is only required for implementing the existing bias mitigation method named ADV. If you do not want to implement this method, you can skip the installation of TensorFlow (the last line of the following commands).
```
pip install sklearn
pip install scikit-opt
pip install numpy
pip install shapely
pip install matplotlib
pip install "tensorflow >= 1.13.1, < 2"
```

## Dataset

We use the four default datasets supported by the AIF360 toolkit. When running the scripts that invoke these datasets, you will be prompted how to download these datasets and in which folders they need to be placed. You can also refer to https://github.com/Trusted-AI/AIF360/tree/master/aif360/data for the raw data files.

## Reproduction of CoT
(1) We obtain the ML performance and fairness metric in protecting a single sensitive feature by our approach CoT (`CoT_Phi.py`). `CoT_Phi.py` supports three arguments: `-d` configures the dataset; `-c` configures the ML algorithm; `-p` configures the protected feature.

```
cd CoT_Replication_Package
python CoT_Phi.py -d adult -c lr -p sex
python CoT_Phi.py -d adult -c lr -p race
python CoT_Phi.py -d compas -c lr -p sex
python CoT_Phi.py -d compas -c lr -p race
python CoT_Phi.py -d german -c lr -p sex
python CoT_Phi.py -d bank -c lr -p age
python CoT_Phi.py -d adult -c rf -p sex
python CoT_Phi.py -d adult -c rf -p race
python CoT_Phi.py -d compas -c rf -p sex
python CoT_Phi.py -d compas -c rf -p race
python CoT_Phi.py -d german -c rf -p sex
python CoT_Phi.py -d bank -c rf -p age
python CoT_Phi.py -d adult -c svm -p sex
python CoT_Phi.py -d adult -c svm -p race
python CoT_Phi.py -d compas -c svm -p sex
python CoT_Phi.py -d compas -c svm -p race
python CoT_Phi.py -d german -c svm -p sex
python CoT_Phi.py -d bank -c svm -p age
```

(2) We obtain the ML performance and fairness metric in protecting multiple sensitive features by our approach CoT (`CoT_Phi_multi.py`). `CoT_Phi_multi.py` supports four arguments: `-d` configures the dataset; `-c` configures the ML algorithm; `-p1` configures the protected feature 1; `-p2` configures the protected feature 2.
```
cd CoT_Replication_Package
python CoT_Phi_multi.py -d adult -c rf -p sex -p2 race
python CoT_Phi_multi.py -d compas -c rf -p sex -p2 race
```

(3) We obtain the ML performance and fairness metric with different multi-objective optimization by our approach CoT (`CoT_Phi_moo_weight.py`). `CoT_Phi_moo_weight.py` supports five arguments: `-d` configures the dataset; `-c` configures the ML algorithm; `-p` configures the protected feature; `-w1` configures the performance weight; `-w2` configures the fairness weight.
```
cd CoT_Replication_Package
python CoT_Phi_moo_weight.py -d adult -c lr -p sex -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d adult -c lr -p race -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d compas -c lr -p sex -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d compas -c lr -p race -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d german -c lr -p sex -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d bank -c lr -p age -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d adult -c rf -p sex -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d adult -c rf -p race -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d compas -c rf -p sex -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d compas -c rf -p race -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d german -c rf -p sex -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d bank -c rf -p age -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d adult -c svm -p sex -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d adult -c svm -p race -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d compas -c svm -p sex -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d compas -c svm -p race -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d german -c svm -p sex -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d bank -c svm -p age -w1 30 -w2 1
python CoT_Phi_moo_weight.py -d adult -c lr -p sex -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d adult -c lr -p race -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d compas -c lr -p sex -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d compas -c lr -p race -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d german -c lr -p sex -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d bank -c lr -p age -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d adult -c rf -p sex -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d adult -c rf -p race -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d compas -c rf -p sex -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d compas -c rf -p race -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d german -c rf -p sex -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d bank -c rf -p age -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d adult -c svm -p sex -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d adult -c svm -p race -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d compas -c svm -p sex -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d compas -c svm -p race -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d german -c svm -p sex -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d bank -c svm -p age -w1 20 -w2 1
python CoT_Phi_moo_weight.py -d adult -c lr -p sex -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d adult -c lr -p race -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d compas -c lr -p sex -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d compas -c lr -p race -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d german -c lr -p sex -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d bank -c lr -p age -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d adult -c rf -p sex -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d adult -c rf -p race -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d compas -c rf -p sex -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d compas -c rf -p race -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d german -c rf -p sex -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d bank -c rf -p age -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d adult -c svm -p sex -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d adult -c svm -p race -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d compas -c svm -p sex -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d compas -c svm -p race -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d german -c svm -p sex -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d bank -c svm -p age -w1 10 -w2 1
python CoT_Phi_moo_weight.py -d adult -c lr -p sex -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d adult -c lr -p race -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d compas -c lr -p sex -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d compas -c lr -p race -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d german -c lr -p sex -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d bank -c lr -p age -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d adult -c rf -p sex -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d adult -c rf -p race -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d compas -c rf -p sex -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d compas -c rf -p race -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d german -c rf -p sex -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d bank -c rf -p age -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d adult -c svm -p sex -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d adult -c svm -p race -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d compas -c svm -p sex -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d compas -c svm -p race -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d german -c svm -p sex -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d bank -c svm -p age -w1 1 -w2 2
python CoT_Phi_moo_weight.py -d adult -c lr -p sex -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d adult -c lr -p race -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d compas -c lr -p sex -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d compas -c lr -p race -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d german -c lr -p sex -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d bank -c lr -p age -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d adult -c rf -p sex -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d adult -c rf -p race -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d compas -c rf -p sex -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d compas -c rf -p race -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d german -c rf -p sex -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d bank -c rf -p age -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d adult -c svm -p sex -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d adult -c svm -p race -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d compas -c svm -p sex -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d compas -c svm -p race -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d german -c svm -p sex -w1 1 -w2 10
python CoT_Phi_moo_weight.py -d bank -c svm -p age -w1 1 -w2 10

```
## Reproduction of existing methods
The reproduction of existing methods can refer to [here](https://github.com/chenzhenpeng18/FSE22-MAAT), where the first author of [MAAT](https://doi.org/10.5522/04/21120121.v1) released their source code of MAAT and replication code of REW, ROC, ADV, Fairway and Fair-SMOTE. We used their code to reproduce existing methods as well.

## Declaration
Thanks to the authors of existing bias mitigation methods for open source, to facilitate our implementation of this paper. Therefore, when using our code or data for your work, please also consider citing their papers, including [AIF360](https://arxiv.org/abs/1810.01943), [Fairway](https://doi.org/10.1145/3368089.3409697), [Fair-SMOTE](https://doi.org/10.1145/3468264.3468537), [Fairea](https://doi.org/10.1145/3468264.3468565) and [MAAT](https://doi.org/10.5522/04/21120121.v1) 


