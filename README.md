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
