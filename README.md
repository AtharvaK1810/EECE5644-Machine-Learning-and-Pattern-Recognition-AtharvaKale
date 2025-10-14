# EECE5644 – Machine Learning and Pattern Recognition  
**Northeastern University**  
**Author:** Atharva Prashant Kale  
**Student ID:** 002442878  
**Email:** kale.ath@northeastern.edu  

---

## 📘 Course Overview
This repository contains all assignments and project work completed for **EECE5644: Machine Learning and Pattern Recognition**, offered by the Department of Electrical and Computer Engineering at Northeastern University.

Each assignment explores different statistical and algorithmic approaches to machine learning through theoretical derivations, MATLAB implementations, and empirical evaluations.  
All results, figures, and analyses follow the IEEE technical paper format.

---

## 📂 Repository Structure
---

## 🧠 Assignment 1 – Gaussian Generative Models

**Topics Covered:**
- Empirical Risk Minimization (ERM) for binary Gaussian classification  
- Naive Bayes approximation with diagonal covariance  
- Fisher Linear Discriminant Analysis (LDA)  
- 4-class Gaussian mixture classification (MAP and ERM rules)  
- Quadratic Discriminant Analysis (QDA) on real-world datasets  
  - Wine Quality dataset (UCI)  
  - Human Activity Recognition (HAR) dataset (UCI)  
- Regularization, PCA visualization, and confidence interval analysis  

**Highlights:**
- Achieved 97% accuracy for 4-class Gaussian MAP classification  
- 99.7% accuracy on HAR dataset using regularized QDA  
- Full ROC analysis and risk minimization with cost-sensitive losses  
- All figures include digital signatures for author attribution  

**Report Location:**  
[Assignment1_GaussianModels/Report/EECE5644_A1_Report.pdf](Assignment1_GaussianModels/Report/EECE5644_A1_Report.pdf)

---

## ⚙️ How to Reproduce Results

1. Open MATLAB and navigate to the Assignment 1 directory:
   cd 'Assignment1_GaussianModels/Matlab'
   run_all
2. Output figures will be saved under `../Figures/`  
3. CSV results (confusion matrices, accuracies, risks) will be saved under `../Results/`  
4. Ensure dataset paths are correctly defined in `run_all.m`:
   - Wine Dataset → `winequality-white.csv`
   - HAR Dataset → `UCI HAR Dataset`

---

## 📚 References
1. C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.  
2. R. O. Duda, P. E. Hart, and D. G. Stork, *Pattern Classification*, 2nd ed. Wiley-Interscience, 2001.  
3. UCI Machine Learning Repository: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
4. UCI Machine Learning Repository: [Human Activity Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

---

## 🧾 License
This repository is provided for academic and educational use.  
© 2025 Atharva Prashant Kale. All rights reserved.
