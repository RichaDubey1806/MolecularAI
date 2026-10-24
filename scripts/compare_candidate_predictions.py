
from pathlib import Path
import sys
import json
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.predict import calculate_features, validate_smiles

MODELS_DIR = PROJECT_ROOT / "models"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Load both models separately. Neither model file is modified.
baseline_model = joblib.load(
    MODELS_DIR / "solubility_rf_baseline.joblib"
)

candidate_model = joblib.load(
    MODELS_DIR / "solubility_rf_clean_candidate.joblib"
)

with open(
    MODELS_DIR / "model_config.json", encoding="utf-8"
) as file:
    baseline_config = json.load(file)

with open(
    MODELS_DIR / "solubility_rf_clean_candidate_config.json",
    encoding="utf-8"
) as file:
    candidate_config = json.load(file)

print("BASELINE MODEL")
print("Trees:", baseline_config.get("n_estimators"))
print("Training samples:", baseline_config.get("training_samples"))

print("\nTUNED CANDIDATE MODEL")
print("Training samples:", candidate_config.get("training_samples"))
print("Minimum samples per leaf:", candidate_config.get("min_samples_leaf"))
print("Max features:", candidate_model.max_features)
print("Expected features:", candidate_config.get("total_features"))
# Same molecules and same feature-generation function for both models.
test_molecules = [
    ("Ethanol", "CCO"),
    ("Acetic acid", "CC(=O)O"),
    ("Benzene", "c1ccccc1"),
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("Caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
    ("Invalid input", "XYZ123"),
]

print("\nPREDICTION COMPARISON")
print("-" * 78)
print(f"{'Molecule':<20} {'Baseline':>12} {'Candidate':>12} {'Difference':>12}")
print("-" * 78)

for name, smiles in test_molecules:
    if not validate_smiles(smiles):
        print(f"{name:<20} {'Invalid SMILES':>38}")
        continue

    features = calculate_features(smiles)

    baseline_prediction = float(baseline_model.predict(features)[0])
    candidate_prediction = float(candidate_model.predict(features)[0])

    difference = candidate_prediction - baseline_prediction

    print(
        f"{name:<20} "
        f"{baseline_prediction:>12.3f} "
        f"{candidate_prediction:>12.3f} "
        f"{difference:>+12.3f}"
    )

print("-" * 78)
print("\nComparison completed. No model files were changed.")
