# EECE 5644 - Machine Learning and Pattern Recognition  
### Assignment 3 (MLP and GMM Experiments)

**Author:** Atharva Prashant Kale  
**NUID:** 002442878  
**Email:** kale.ath@northeastern.edu  
**Date:** November 2025  

---

## 📘 Overview
This repository contains the code, figures, and report for **Assignment 3** of *EECE 5644 – Machine Learning and Pattern Recognition* at Northeastern University.  
The assignment includes two main experiments:

1. **Question 1 – MLP Classifier:** Training a 4-class neural network on 3D Gaussian data and comparing test error against the Bayes-optimal classifier.  
2. **Question 2 – GMM Model Order Selection:** Using 10-fold cross-validation to determine the correct number of Gaussian mixture components for overlapping clusters.

Work was completed in **Python 3.10 (Google Colab)** and **MATLAB R2025b**, following TA feedback regarding activation type, dimensionality, and cross-validation folds.

---

## 🧠 Question 1 – Multi-Layer Perceptron (MLP)

### Objective
Train and evaluate a single-hidden-layer MLP using ELU activation and softmax output on synthetic 3D Gaussian data.

### Model
z₁ = W₁x + b₁  
h  = ELU(z₁)  
z₂ = W₂h + b₂  
ŷ  = softmax(z₂)

### Experiment Details
- Hidden layer widths tested: P ∈ {4, 8, 16, 32, 64}  
- Data sizes: N ∈ {100, 500, 1000, 5000, 10000}  
- Cross-validation: 10-fold  
- Random restarts per fold: 5  
- Optimizer: SGD with exponential learning rate decay  
- Oracle classifier computed using the true Gaussian parameters  
- Standardization applied based on training data mean and variance  

### Outputs
- Q1_mlp_results.csv  
- PCA visualization: Q1_pca_scatter.pdf  
- Confusion matrices for each N: Q1_confusion_mlp_test_N*.pdf  
- Semilog plot of test P(error) vs N: Q1_mlp_test_error.pdf  

### Result Summary

| N | P★ | Test Error | Oracle Error |
|:---:|:---:|:---:|:---:|
| 100 | 64 | 0.050 | 0.068 |
| 500 | 64 | 0.080 | 0.068 |
| 1000 | 64 | 0.080 | 0.068 |
| 5000 | 64 | 0.061 | 0.068 |
| 10000 | 64 | 0.071 | 0.068 |

### Observations
- The oracle error (6.8%) lies slightly below the assignment target of 10–20%, due to low covariance overlap.  
- The 5% test error at N=100 is marginally below the oracle value, likely caused by small-sample variance rather than an actual performance gain.  
- As N increases, the error stabilizes near the oracle limit, confirming convergence.  

---

## 🔢 Question 2 – Gaussian Mixture Model Selection

### Objective
Identify the correct number of Gaussian components using 10-fold cross-validation and log-likelihood maximization.

### Data Setup
- True distribution: 4 components with two overlapping clusters  
- Dataset sizes: N = {10, 100, 1000}  
- Candidate models: K = 1 … 10  
- Repetitions: 100  
- Evaluation metric: Mean log-likelihood (per sample)  
- Library: sklearn.mixture.GaussianMixture  

### Outputs
- Q2_gmm_selection_rates_py.csv  
- Example mixture scatter: Q2_example_fit_py.png  
- CV selection heatmap: Q2_gmm_selection_heatmap_py.png  
- Selection rate bar plots: Q2_selection_bars_N10_py.png, Q2_selection_bars_N100_py.png, Q2_selection_bars_N1000_py.png  

### Key Trends
- For N = 10, underfitting occurs (K = 1 dominates).  
- For N = 100, model correctly favors K = 3, close to the true structure.  
- For N = 1000, stability around K = 3–4 shows accurate convergence.  
- Larger N improves consistency and reduces overfitting variance.  

---

## ⚙️ How to Run (Colab One-Click)

1. Open Google Colab (https://colab.research.google.com/)  
2. Upload `assignment3_oneclick.py` or paste its contents into a new notebook  
3. Run all cells  
4. Outputs are saved automatically under:  

a3_outputs/  
 ├── figs/  
 └── results/  

**Runtime:** Q1 ≈ 10 minutes, Q2 ≈ 25 minutes (on Colab T4 GPU)

---

## 🧩 Directory Structure
EECE5644_A3/  
│  
├── a3_outputs/  
│ ├── figs/  
│ │ ├── Q1_*.pdf  
│ │ └── Q2_*.png  
│ └── results/  
│  ├── Q1_mlp_results.csv  
│  └── Q2_gmm_selection_rates_py.csv  
│  
├── assignment3_oneclick.py  
├── EECE5644_Assignment3_Report.pdf  
└── README.md  

---

## 🧾 References
1. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer (2006)  
2. K. P. Murphy, *Machine Learning: A Probabilistic Perspective*, MIT Press (2012)  
3. I. Goodfellow et al., *Deep Learning*, MIT Press (2016)  
4. scikit-learn v1.5 documentation  
5. PyTorch v2.3 documentation  
6. MATLAB R2025b ML Toolbox documentation  

---

## 🔗 Repository
**GitHub:** https://github.com/atharvakale/EECE5644_Assignment3
