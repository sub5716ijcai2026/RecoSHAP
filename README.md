# Anonymous Project

## All Models Are Wrong, but Some Are Interchangeably Right

This repository contains the code and experiments for a systematic study of explanation agreement across machine learning models using SHAP.

We investigate when different predictive models can be considered *explanation-interchangeable*: that is, when they produce highly similar feature-importance rankings despite having very different architectures, training procedures, and computational costs.

### Problem Statement

Our goal is to quantify when models are interchangeable in terms of explanations, and to provide operational guidance for selecting interpretable and computationally efficient surrogate models without sacrificing explanatory reliability.

### Approach

We analyze SHAP-based feature importance rankings across a broad set of regression and classification models, including:

- Linear models  
- Tree-based methods  
- Ensemble methods  
- Neural networks  

Model alignment is measured using Normalized Discounted Cumulative Gain (NDCG), a standard ranking similarity metric that emphasizes agreement among top-ranked features.

### Key Contributions

- A large-scale cross-model comparison of SHAP explanations  
- A formal definition of **explanation interchangeability** based on high-consensus thresholds  
- Empirical evidence that lightweight models often provide explanations nearly identical to those of complex ensembles  
- A practical framework for selecting a “centroid” model that balances interpretability, robustness, and efficiency  

### Repository Structure

