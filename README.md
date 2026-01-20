# Anonymous Project: RecoSHAP

**All Models Are Wrong, but Some Are Interchangeably Right**

This repository contains the code and experiments for a systematic study of explanation agreement across machine learning models using SHAP.

---

## Problem Statement

We investigate when predictive models can be considered **explanation-interchangeable**, i.e., when different models produce highly similar feature-importance rankings despite varying architectures, training procedures, and computational costs.

Our goal is to provide operational guidance for selecting interpretable and computationally efficient surrogate models **without sacrificing explanatory reliability**.

---

## Approach

- Analyze **SHAP-based feature importance rankings** across a broad set of models, including:
  - Linear models
  - Tree-based methods
  - Ensemble methods
  - Neural networks
- Measure model alignment using **Normalized Discounted Cumulative Gain (NDCG)**, which emphasizes agreement among top-ranked features.
- Quantify **global vs local importance patterns** to identify surrogate models that best capture system-level tendencies.

## Datasets Description

This project uses a variety of classification and regression datasets to evaluate explanation interchangeability across models.

### Classification Datasets
### Classification Datasets

| Dataset        | Instances (m) | Features (d) | Classes |
|----------------|---------------|--------------|---------|
| iris           | 150           | 5            | 3       |
| wine           | 178           | 14           | 3       |
| breast-cancer  | 286           | 10           | 2       |
| diabetes       | 768           | 9            | 2       |
| vehicle        | 846           | 19           | 4       |
| Schelling      | 1000          | 5            | 2       |
| cmc            | 1473          | 10           | 3       |
| car            | 1728          | 7            | 4       |
| hypothyroid    | 3163          | 26           | 2       |
| chess          | 3196          | 37           | 2       |
| splice         | 3188          | 61           | 3       |
| churn          | 5000          | 21           | 2       |
| Loan_Modelling | 5000          | 13           | 2       |
| mushroom       | 8124          | 23           | 2       |
| Adult          | 48842         | 15           | 2       |
| shuttle        | 58000         | 10           | 5       |

### Regression Datasets

| Dataset           | Instances ($m$) | Features ($d$) | Problem Size |
|------------------|----------------|----------------|--------------|
| analcatdata_apnea1 | 475           | 3              | small        |
| ERA               | 44            | 4              | small        |
| LEV               | 92            | 4              | small        |
| ESL               | 199           | 4              | small        |
| pm10              | 500           | 7              | medium       |
| pollen            | 3,848         | 4              | medium       |
| Wine_Quality      | 1,018         | 11             | medium       |
| Abalone           | 4,177         | 8              | big          |
| puma8NH           | 8,192         | 8              | big          |
| cpu_small         | 8,192         | 12             | big          |
| wind              | 6,574         | 14             | big          |
| satellite_image   | 6,435         | 36             | big          |
| pol               | 14,958        | 48             | very big     |
| houses            | 20,640        | 8              | very big     |
| BNG_lowbwt        | 31,104        | 9              | very big     |
| Schelling         | 1000          | 5              | -            |

---

## Key Contributions

1. Large-scale cross-model comparison of SHAP explanations  
2. Formal definition of **explanation interchangeability** based on high-consensus thresholds  
3. Empirical evidence that **lightweight models** often provide explanations nearly identical to complex ensembles  
4. Practical framework to select a **centroid model** balancing interpretability, robustness, and computational efficiency

---

## Repository Structure

Figures/ # Plots and visualizations

Results/ # Key output files

Run_tests/ # Scripts to run experiments

Schelling_ABS/ # Schelling model experiments

post_procs/ # Post-processing scripts

XAI_and_reco_SHAP-full_paper.pdf # Main paper

XAI_and_reco_SHAP-Supplementary_materials.pdf # Supplementary materials


---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/sub5716ijcai2026/RecoSHAP.git
cd RecoSHAP

    Install dependencies (if using Python):

pip install -r requirements.txt

    Run experiments or analysis scripts in Run_tests/ and inspect results in Results/.

License

This project is licensed under the BSD-3-Clause License
.
