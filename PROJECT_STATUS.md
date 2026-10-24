# MolecularAI — Project Status

## Project Overview

MolecularAI is a machine-learning application for predicting aqueous
molecular solubility from SMILES representations.

The application uses RDKit for molecular processing and descriptor
calculation, a trained Random Forest model for prediction, and Streamlit
for the user interface.

## Stage 0 — Environment and Project Setup

Status: Complete

- Python virtual environment and scientific libraries configured
- RDKit installed and tested
- Git repository initialized
- Project structure and documentation created

## Stage 1 — Machine Learning Foundations

Status: Complete

Topics studied:
- Linear regression, weights, bias and gradient descent
- Mean squared error and model optimization
- Train, validation and test splits
- Underfitting, overfitting and model complexity
- Feature scaling and numerical stability
- MAE, MSE, RMSE and R-squared
- Comparing regression models

## Stage 2 — Molecular Data and Feature Engineering

Status: Complete for the current implementation

- Loaded the Delaney ESOL molecular solubility dataset
- Processed molecular SMILES using RDKit
- Canonicalized molecular structures
- Aggregated repeated structures by mean target value
- Excluded invalid molecular structures
- Calculated six molecular descriptors
- Generated Morgan fingerprints with radius 2 and 2048 bits
- Combined descriptors and fingerprints into 2054 model features

## Stage 3 — Model Training and Evaluation

Status: Implemented; further independent validation remains

- Built Random Forest regression models
- Evaluated models using scaffold-based data splits
- Compared configurations and analyzed prediction errors
- Generated evaluation tables and diagnostic plots
- Documented data leakage safeguards and evaluation limitations
- Trained the candidate model on 1117 cleaned unique molecular structures

Candidate model configuration:
- Algorithm: RandomForestRegressor
- Number of trees: 200
- Maximum depth: 12
- Minimum samples per leaf: 2
- Number of features: 2054

Development evaluation results for a separately trained evaluation
configuration:
- MAE: 0.5920
- RMSE: 0.7447
- R-squared: 0.7688

These results came from development scaffold splits used during model
selection. They are not an untouched final benchmark and are not direct
test metrics for the saved candidate model.

The saved candidate model was refitted using all 1117 cleaned molecules.
Its training predictions are not independent estimates of generalization.
External validation has not yet been performed.

## Stage 4 — Streamlit Application

Status: Implemented and manually tested

Single Molecule mode:
- Accepts a molecular SMILES string
- Predicts log aqueous solubility
- Displays a 2D molecular structure
- Shows six molecular descriptors

Compare Molecules mode:
- Accepts two molecular SMILES strings
- Predicts both molecules
- Displays descriptor tables
- Provides a comparison summary and visual charts

Manual checks passed for:
- Ethanol: CCO
- Acetic acid: CC(=O)O
- Benzene: c1ccccc1

The displayed values are model estimates, not experimental measurements.

## Stage 5 — Documentation and Portfolio Preparation

Status: In progress

Next tasks:
1. Finalize the README and project setup instructions.
2. Review the model card and evaluation reports.
3. Check repository cleanliness and tracked files.
4. Verify that large model binaries and raw datasets are not committed.
5. Add clear screenshots of the application when available.
6. Document limitations and future improvements.
7. Plan independent validation on data not used for model selection.

## Important Scientific Limitations

- Predictions depend on the dataset and molecular structures represented
  during training.
- Development split metrics may be optimistic after repeated model tuning.
- Independent external validation is still required.
- Predictions should not be treated as experimental measurements or as
  evidence of drug safety or efficacy.

## Learning Principle

Understand the concept -> derive the logic -> implement -> test ->
use established libraries -> apply to molecular data.
