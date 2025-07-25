# Leveraging Molecular Fingerprints and DFT Inputs for pKa Prediction Using XGBoost

Authors: Kyle Liang, Harry Park and Amitesh Senthilkumar

Mentors: Dr. Andreas Goetz and Dr. Vikrant Tripathy 

REHS Program, San Diego Supercomputer Center, UCSD, San Diego, CA, 92093

Original Paper/Methodology Credit: Sanchez et al. (2024) https://github.com/sarmaier/RootedCBH_pka

### Abstract
> Predicting pKa values accurately remains a key challenge in computational chemistry. Previous work has shown that combining low-cost DFT calculations with machine learning — specifically random forests using CBH-derived features — improves performance. However, such models may underutilize structural information.  This study investigates utilizing Gradient Boosting (XGBoost) models using the same DFT features. 

Link to research poster: 

<img width="460" alt="image" src="https://github.com/sarmaier/RootedCBH_pka/assets/152440946/e996ff23-1e6f-45e1-8757-575d4b3a82d5">

### Original Random Forest Framework (by Sanchez et al.)

RootedCBH_pka repository contains tools to run QM/ML (random forest) framework for the accurate prediction of pKas of complex organic molecules using physics-based features from DFT and structural features from our CBH fragmentation protocol. Our model corrects the functional group specific deficiencies associated with DFT and achieves impressive accuracy on two external benchmark test sets, the SAMPL6 challenge and Novartis datasets.

Link to paper: https://pubs.acs.org/doi/10.1021/acs.jcim.3c01923

## Requirements

* pandas~=1.0.1

* numpy~=1.19.5

* networkx~=2.5.1

* rdkit~=2020.03.3.0

* scipy~=1.5.3

* scikit-learn~=0.24.2

* json-numpy-1.0.1



## Citation
Sanchez, A. J.; Maier, S.; Raghavachari, K. Leveraging DFT and Molecular Fragmentation for Chemically Accurate p K a Prediction Using Machine Learning. J. Chem. Inf. Model. 2024, acs.jcim.3c01923. https://doi.org/10.1021/acs.jcim.3c01923



