# MolecularAI — AI-Powered Molecular Property Prediction

MolecularAI is a machine-learning application that predicts the aqueous solubility of small molecules from their SMILES representations. It combines cheminformatics, molecular descriptors, molecular fingerprints, and a Random Forest regression model in an interactive Streamlit interface.

## Features

- Predict molecular solubility from SMILES input.
- Compare multiple molecules side by side.
- Visualize molecular structures using RDKit.
- Calculate six molecular descriptors:
  - Molecular weight
  - Hydrogen-bond donors
  - Hydrogen-bond acceptors
  - Rotatable bonds
  - Topological polar surface area (TPSA)
  - LogP
- Display molecular property comparisons and visualizations.
- Evaluate model performance with scaffold-based data splits and error-analysis reports.

## Technology Stack

- Python
- Streamlit
- RDKit
- scikit-learn
- NumPy and pandas
- Matplotlib
- Joblib

## Machine Learning Approach

The candidate model is a Random Forest regressor trained using the ESOL aqueous-solubility dataset.

The feature representation contains 2,054 features:
- 6 molecular descriptors
- 2,048 Morgan fingerprint bits, using radius 2

Candidate model configuration:
- Estimator: RandomForestRegressor
- Number of trees: 200
- Maximum depth: 12
- Minimum samples per leaf: 2
- Random state: 42

The cleaned training data contains 1,117 unique canonical molecular structures after repeated structures are aggregated and invalid SMILES are excluded.

## Run Locally

### 1. Clone the repository

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd MolecularAI
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Add the trained model files

The model artifacts are not included in this repository by default. Place the required candidate model and configuration files in the `models/` directory:

- `solubility_rf_clean_candidate.joblib`
- `solubility_rf_clean_candidate_config.json`

These files are required to run candidate-model predictions.

### 5. Launch the application

In PowerShell:

```powershell
$env:MOLECULARAI_MODEL = "candidate"
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed in the terminal.

## Example SMILES

| Molecule | SMILES |
|---|---|
| Ethanol | `CCO` |
| Acetic acid | `CC(=O)O` |
| Benzene | `c1ccccc1` |

## Evaluation

During development, a separately trained Random Forest model was evaluated using a cleaned scaffold-based split:

| Metric | Development result |
|---|---:|
| MAE | 0.5920 |
| RMSE | 0.7447 |
| R² | 0.7688 |

These results describe the evaluation model and development split, not an independent benchmark of the saved candidate model. The split was used during model development and configuration selection. The saved candidate model was subsequently refitted on all 1,117 cleaned molecules. Its training performance is not an independent estimate of generalization, and external validation has not yet been performed.

See `reports/MODEL_CARD.md` for model details, evaluation context, limitations, and intended use.

## Limitations

- Predictions are estimates, not experimental measurements.
- Performance may vary for molecules unlike those represented in the training dataset.
- Development-set metrics should not be interpreted as external validation.
- Predictions should not be used alone to make medical, clinical, or drug-development decisions.

## Project Status

The interactive application supports single-molecule prediction and molecule comparison. Further work can include independent validation, uncertainty estimation, broader datasets, and additional molecular visualization capabilities.

## License

No license has been specified yet. Add a license before explicitly permitting reuse or redistribution.
