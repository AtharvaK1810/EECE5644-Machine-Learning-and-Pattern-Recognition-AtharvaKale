# 🧠 EECE5644 – Assignment 1: Gaussian Generative Models  
**Course:** Machine Learning and Pattern Recognition  
**University:** Northeastern University  
**Author:** Atharva Prashant Kale (002442878)  
**Email:** kale.ath@northeastern.edu  

---

## 📘 Overview
This assignment explores Bayesian decision theory and Gaussian generative classifiers through simulation, analytical derivations, and real-world data.  
The work demonstrates the performance trade-offs among **ERM**, **Naive Bayes**, **LDA**, and **QDA**, as well as their extensions to multiclass and cost-sensitive problems.

---

## 📂 Structure
Assignment1_GaussianModels/
│
├── Matlab/
│   ├── run_all.m                # Runs all questions sequentially
│   ├── q1_main.m                # ERM, Naive Bayes, LDA
│   ├── q2_main.m                # 4-class MAP and ERM classifiers
│   ├── q3_main.m                # QDA on Wine & HAR datasets
│   └── Utils/                   # Helper functions
│       ├── add_signature.m
│       ├── empirical_error.m
│       ├── log_gaussian_pdf.m
│       ├── predict_gaussian_generative.m
│       ├── fit_gaussian_generative.m
│       └── roc_from_scores.m
│
├── Figures/                     # Auto-generated plots with signatures
├── Results/                     # CSV & text outputs
└── Report/EECE5644_A1_Report.pdf

---

## ⚙️ How to Run

1. Open MATLAB (R2025b or later).  
2. Navigate to the **Matlab** folder:
   ```matlab
   cd 'Assignment1_GaussianModels/Matlab'
   run_all

3.	Figures are saved in ../Figures/ and results in ../Results/.
4.	Datasets for Q3 (Wine Quality, HAR) must be placed in the same directory or have their paths set in the script.

