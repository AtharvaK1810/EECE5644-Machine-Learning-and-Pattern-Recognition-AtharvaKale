# EECE5644 Assignment 4  
SVM, MLP, and GMM Based Image Segmentation  
Author: Atharva Prashant Kale  
NUID: 002442878  
Email: kale.ath@northeastern.edu  

---

## Overview
This repository contains all code, output figures, and documentation for Assignment 4 of EECE5644 Machine Learning and Pattern Recognition.  
The assignment covers two major tasks:

1. **Nonlinear binary classification** using  
   - Support Vector Machine with an RBF kernel  
   - Multilayer Perceptron with one hidden layer  

2. **Unsupervised image segmentation** using  
   - Gaussian Mixture Models with spatial and color features  
   - Model order selection using cross validation  

All experiments were performed using a single integrated Python script designed for Google Colab.

---

## Directory Structure

```
Assignment 4/
│
├── a4_script.py                 # Full Colab compatible script (SVM, MLP, GMM segmentation)
├── a4_outputs/
│   ├── q1_figs/                 # Heatmaps and decision boundaries for SVM and MLP
│   ├── q2_figs/                 # GMM CV curve and segmentation results
│
├── Assignment 4 Report.pdf      # Full written report
├── README.md                    # This file
```

---

## How to Run the Code

### 1. Open Google Colab
Go to https://colab.research.google.com/

### 2. Create a new notebook and upload `a4_script.py`
Click File > Upload Notebook or copy and paste the entire script into a single code cell.

### 3. Run the script
In Colab, the only command you need is:

```python
!python a4_script.py
```

The script automatically:
- Generates synthetic ring data  
- Runs SVM and MLP with 10 fold cross validation  
- Downloads the BSDS300 dataset  
- Builds 5D pixel features  
- Runs GMM model order selection  
- Creates all segmentation figures  

All results are saved inside:

```
a4_outputs/
```

---

## Output Figures

The script produces **7 figures**, saved under:

- `a4_outputs/q1_figs/`
  - `q1_svm_cv_heatmap.png`
  - `q1_svm_boundary.png`
  - `q1_mlp_cv_curve.png`
  - `q1_mlp_boundary.png`

- `a4_outputs/q2_figs/`
  - `q2_original_image.png`
  - `q2_gmm_cv_loglik_vs_K.png`
  - `q2_segmentation_side_by_side.png`

Every figure includes a signature:
```
Atharva Prashant Kale | NUID: 002442878 | kale.ath@northeastern.edu
```

---

## Notes
- The BSDS300 dataset is downloaded automatically during execution.  
- The script is seed controlled for reproducibility.  
- Gaussian Mixture Models use full covariance matrices to match assignment requirements.  
- Model selection for SVM, MLP, and GMM is performed using cross validation.  

---

## License
This repository is for academic use only.
