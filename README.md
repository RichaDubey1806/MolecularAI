# MolecularAI - AI-Powered Molecular Property Prediction

MolecularAI is a scientific machine learning project that predicts the aqueous solubility of small molecules from their SMILES representations. It combines cheminformatics, molecular descriptors, Morgan fingerprints, and Random Forest regression in an interactive Streamlit application.

The long-term vision is to extend this foundation toward molecular discovery and design, including uncertainty estimation, candidate generation, molecular optimization, and scientific machine learning.

## Current Features

- Predict aqueous solubility from a molecule's SMILES string.
- Compare multiple molecules side by side.
- Visualize 2D molecular structures using RDKit.
- Calculate six molecular descriptors:
  - Molecular weight
  - Hydrogen-bond donors
  - Hydrogen-bond acceptors
  - Rotatable bonds
  - Topological polar surface area (TPSA)
  - LogP
- Display molecular property comparison tables and charts.
- Evaluate models using scaffold-based splits and error-analysis scripts.

## Technology Stack

- **Language:** Python
- **Application:** Streamlit
- **Cheminformatics:** RDKit
- **Machine learning:** scikit-learn
- **Data processing:** NumPy, pandas
- **Visualization:** Matplotlib
- **Model persistence:** Joblib

## Machine Learning Approach

The application uses a Random Forest regressor to estimate aqueous solubility from molecular features.

The feature representation contains 2,054 features:

- 6 molecular descriptors
- 2,048 Morgan fingerprint bits, using radius 2

The candidate model configuration is:

- Estimator: `RandomForestRegressor`
- Number of trees: 200
- Maximum depth: 12
- Minimum samples per leaf: 2
- Random state: 42

The cleaned dataset contains 1,117 unique canonical molecular structures after invalid SMILES are excluded and repeated structures are aggregated.

## Project Architecture

1. **Input:** The user enters a molecule as a SMILES string.
2. **Molecular processing:** RDKit validates the structure and calculates molecular descriptors and Morgan fingerprints.
3. **Prediction:** The trained Random Forest model estimates aqueous solubility.
4. **Visualization:** Streamlit presents predictions, 2D molecular structures, descriptors, and comparison charts.
5. **Evaluation:** Separate scripts support scaffold-based evaluation, model configuration comparisons, and error analysis.

## Repository Structure

```text
MolecularAI/
├── app/
│   ├── predict.py
│   └── streamlit_app.py
├── data/
├── models/
├── notebooks/
├── reports/
│   └── MODEL_CARD.md
├── scripts/
├── requirements.txt
├── README.md
└── PROJECT_STATUS.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/RichaDubey1806/MolecularAI.git
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

### 4. Provide the trained model files

The trained model binaries and raw dataset are excluded from Git. To run predictions, place these candidate model files in the `models/` directory:

- `solubility_rf_clean_candidate.joblib`
- `solubility_rf_clean_candidate_config.json`

### 5. Launch the application

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

During development, a separately trained Random Forest model was evaluated using a cleaned scaffold-based split.

| Metric | Development result |
|---|---:|
| MAE | 0.5920 |
| RMSE | 0.7447 |
| R² | 0.7688 |

These results describe the evaluation model and development split, not an independent benchmark of the saved candidate model. The split was used during model development and configuration selection. The saved candidate model was subsequently refitted on all 1,117 cleaned molecules. Its training performance is not an independent estimate of generalization, and external validation has not yet been performed.

See `reports/MODEL_CARD.md` for further details, evaluation context, limitations, and intended use.

## Limitations

- Predictions are estimates, not experimental measurements.
- Performance may vary for molecules unlike those represented in the training data.
- Development-set metrics should not be interpreted as external validation.
- Predictions should not be used alone to make medical, clinical, or drug-development decisions.

## Future Work

The long-term roadmap includes:

- Independent external validation
- Prediction uncertainty estimation
- Broader molecular datasets
- Molecular representation learning and graph neural networks
- Candidate molecule generation and optimization
- Interactive 3D molecular visualization
- Scientific and physics-informed machine learning

These are future directions, not claims about currently implemented features.

## License

No license has been specified. Add a license before explicitly permitting reuse or redistribution.