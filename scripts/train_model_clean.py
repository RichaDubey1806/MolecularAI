
from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "delaney-processed.csv"
MODELS_DIR = PROJECT_ROOT / "models"

OLD_CONFIG_PATH = MODELS_DIR / "model_config.json"
CANDIDATE_MODEL_PATH = MODELS_DIR / "solubility_rf_clean_candidate.joblib"
CANDIDATE_CONFIG_PATH = MODELS_DIR / "solubility_rf_clean_candidate_config.json"

TARGET_COLUMN = "measured log solubility in mols per litre"
SMILES_COLUMN = "smiles"


# --------------------------------------------------
# 2. Load the existing feature-generation function
# --------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from predict import calculate_features


def main():
    print("=" * 60)
    print("MolecularAI: Clean Candidate Model Training")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not OLD_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Model config not found: {OLD_CONFIG_PATH}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(OLD_CONFIG_PATH, "r", encoding="utf-8") as file:
        old_config = json.load(file)

    # --------------------------------------------------
    # 3. Load and clean the dataset
    # --------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    required_columns = {SMILES_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[[SMILES_COLUMN, TARGET_COLUMN]].copy()
    df[SMILES_COLUMN] = df[SMILES_COLUMN].astype("string").str.strip()
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    df = df.dropna(subset=[SMILES_COLUMN, TARGET_COLUMN])
    df = df[df[SMILES_COLUMN] != ""].copy()

    print(f"Rows after basic cleaning: {len(df)}")

    # --------------------------------------------------
    # 4. Convert SMILES to canonical molecular structures
    # --------------------------------------------------
    canonical_smiles = []
    invalid_count = 0

    for smiles in df[SMILES_COLUMN]:
        molecule = Chem.MolFromSmiles(str(smiles))

        if molecule is None:
            canonical_smiles.append(None)
            invalid_count += 1
        else:
            canonical_smiles.append(
                Chem.MolToSmiles(molecule, canonical=True)
            )

    df["canonical_smiles"] = canonical_smiles
    df = df.dropna(subset=["canonical_smiles"]).copy()

    print(f"Invalid SMILES removed: {invalid_count}")
    print(f"Valid rows before duplicate aggregation: {len(df)}")

    # --------------------------------------------------
    # 5. Aggregate repeated measurements of the same molecule
    # --------------------------------------------------
    original_valid_rows = len(df)

    clean_df = (
        df.groupby("canonical_smiles", as_index=False)
        .agg(
            **{
                TARGET_COLUMN: (TARGET_COLUMN, "mean"),
                "measurement_count": (TARGET_COLUMN, "size"),
            }
        )
    )

    print(f"Unique canonical molecules: {len(clean_df)}")
    print(
        "Rows combined during duplicate aggregation: "
        f"{original_valid_rows - len(clean_df)}"
    )

    # --------------------------------------------------
    # 6. Generate the same 2054 features used by the app
    # --------------------------------------------------
    print("\nGenerating molecular features...")

    feature_rows = []

    for index, smiles in enumerate(clean_df["canonical_smiles"], start=1):
        features = calculate_features(smiles)

        if features is None:
            raise ValueError(
                f"Feature generation failed for molecule: {smiles}"
            )

        feature_rows.append(features[0])

        if index % 100 == 0 or index == len(clean_df):
            print(f"Processed {index}/{len(clean_df)} molecules")

    X = pd.DataFrame(feature_rows).to_numpy(dtype=float)
    y = clean_df[TARGET_COLUMN].to_numpy(dtype=float)

    expected_features = int(old_config["total_features"])

    if X.shape != (len(clean_df), expected_features):
        raise ValueError(
            f"Unexpected feature matrix shape: {X.shape}; "
            f"expected ({len(clean_df)}, {expected_features})"
        )

    if not (pd.notna(X).all() and pd.notna(y).all()):
        raise ValueError("Features or targets contain missing values.")

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    # --------------------------------------------------
    # 7. Train the candidate Random Forest model
    # --------------------------------------------------
    print("\nTraining candidate model...")

    model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
)

    model.fit(X, y)

    # --------------------------------------------------
    # 8. Save as NEW files; preserve the existing model
    # --------------------------------------------------
    joblib.dump(model, CANDIDATE_MODEL_PATH)

    candidate_config = old_config.copy()
    candidate_config.update(
        {
            "project": "MolecularAI",
            "model_type": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "random_state": 42,
            "training_samples": int(len(clean_df)),
            "total_features": int(X.shape[1]),
            "feature_type": (
                "molecular_descriptors_plus_morgan_fingerprints"
            ),
            "training_data_cleaning": (
                "Canonical SMILES; duplicate structures aggregated "
                "using mean measured solubility"
            ),
            "validation_note": (
                "Candidate refitted on all cleaned dataset molecules "
                "after exploratory repeated scaffold validation. "
                "Training performance is not an independent test result; "
                "external validation has not been performed."
            ),
        }
    )

    with open(CANDIDATE_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(candidate_config, file, indent=4)

    # --------------------------------------------------
    # 9. Print a verification summary
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("CANDIDATE MODEL TRAINING COMPLETED")
    print("=" * 60)
    print(f"Model type: {type(model).__name__}")
    print(f"Training molecules: {len(clean_df)}")
    print(f"Features per molecule: {X.shape[1]}")
    print(f"Trees: {model.n_estimators}")
    print(f"Model saved to: {CANDIDATE_MODEL_PATH}")
    print(f"Config saved to: {CANDIDATE_CONFIG_PATH}")

    print("\nExisting app model was NOT overwritten.")
    print("No training-set score is reported as test performance.")


if __name__ == "__main__":
    main()
