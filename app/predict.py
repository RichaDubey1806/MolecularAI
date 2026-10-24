
from pathlib import Path
import json
import os
import joblib

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Keep the original model as the default.
MODEL_VARIANT = os.environ.get(
    "MOLECULARAI_MODEL", "baseline"
).strip().lower()

MODEL_FILES = {
    "baseline": (
        MODELS_DIR / "solubility_rf_baseline.joblib",
        MODELS_DIR / "model_config.json",
    ),
    "candidate": (
        MODELS_DIR / "solubility_rf_clean_candidate.joblib",
        MODELS_DIR / "solubility_rf_clean_candidate_config.json",
    ),
}

if MODEL_VARIANT not in MODEL_FILES:
    raise ValueError(
        "Invalid MOLECULARAI_MODEL. Use 'baseline' or 'candidate'."
    )

MODEL_PATH, CONFIG_PATH = MODEL_FILES[MODEL_VARIANT]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Model configuration not found: {CONFIG_PATH}")

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)


def get_model_info():
    return {
        "model_variant": MODEL_VARIANT,
        "model_file": MODEL_PATH.name,
        "model_type": type(model).__name__,
        "expected_features": config["total_features"],
        "prediction_task": config["task"],
    }


def validate_smiles(smiles):
    """Check whether RDKit can parse the supplied SMILES string."""
    if not isinstance(smiles, str) or not smiles.strip():
        return False

    molecule = Chem.MolFromSmiles(smiles.strip())
    return molecule is not None


# Use the same fingerprint settings as during model training.
fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=config["fingerprint_radius"],
    fpSize=config["fingerprint_bits"],
)


def calculate_features(smiles):
    """Convert a valid SMILES string into the model's features."""
    molecule = Chem.MolFromSmiles(smiles.strip())

    if molecule is None:
        raise ValueError("Invalid SMILES string.")

    # Six descriptors, in the training order.
    descriptors = np.array(
        [
            Descriptors.MolWt(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Lipinski.NumRotatableBonds(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Descriptors.MolLogP(molecule),
        ],
        dtype=float,
    )

    # Generate the Morgan fingerprint.
    fingerprint = fingerprint_generator.GetFingerprint(molecule)

    fingerprint_array = np.zeros(
        config["fingerprint_bits"], dtype=np.int8
    )
    DataStructs.ConvertToNumpyArray(
        fingerprint, fingerprint_array
    )

    # Descriptors first, fingerprints second.
    features = np.concatenate([descriptors, fingerprint_array])

    if len(features) != config["total_features"]:
        raise ValueError(
            "Generated feature count does not match model configuration."
        )

    return features.reshape(1, -1)


def predict_solubility(smiles):
    """Predict aqueous solubility from a SMILES string."""
    if not validate_smiles(smiles):
        raise ValueError(
            "Invalid SMILES string. Please check your input."
        )

    features = calculate_features(smiles)
    prediction = model.predict(features)[0]

    return float(prediction)


if __name__ == "__main__":
    print("MolecularAI prediction engine loaded!")
    print("Model information:", get_model_info())

    test_molecules = [
        ("Ethanol", "CCO"),
        ("Acetic acid", "CC(=O)O"),
        ("Benzene", "c1ccccc1"),
        ("Invalid input", "XYZ123"),
    ]

    print("\nMolecule prediction tests:")

    for name, smiles in test_molecules:
        try:
            features = calculate_features(smiles)
            prediction = predict_solubility(smiles)
            print(
                f"{name}: prediction={prediction:.3f}, "
                f"feature_shape={features.shape}"
            )
        except ValueError:
            print(f"{name}: Invalid SMILES rejected")
