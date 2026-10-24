
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


# Project paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from predict import calculate_features  # noqa: E402


DATA_PATH = ROOT / "data" / "raw" / "delaney-processed.csv"
PREDICTIONS_PATH = ROOT / "reports" / "scaffold_test_predictions.csv"
OUTPUT_PATH = ROOT / "reports" / "training_coverage_analysis.csv"

SMILES_COLUMN = "smiles"
TARGET = "measured log solubility in mols per litre"

DESCRIPTOR_NAMES = [
    "molecular_weight",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "TPSA",
    "LogP",
]


def get_scaffold(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold)


def get_fingerprint(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius=2,
        nBits=2048,
    )


def get_descriptors(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return [
        Descriptors.MolWt(molecule),
        Lipinski.NumHDonors(molecule),
        Lipinski.NumHAcceptors(molecule),
        Lipinski.NumRotatableBonds(molecule),
        rdMolDescriptors.CalcTPSA(molecule),
        Descriptors.MolLogP(molecule),
    ]


def main():
    print("Loading dataset and previous test predictions...")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[SMILES_COLUMN, TARGET]).copy()
    df = df.drop_duplicates(subset=[SMILES_COLUMN]).reset_index(drop=True)

    # Reproduce the EXACT split from evaluate_model.py.
    df["scaffold"] = df[SMILES_COLUMN].apply(get_scaffold)

    rng = np.random.default_rng(42)
    scaffold_groups = list(df.groupby("scaffold").groups.values())
    rng.shuffle(scaffold_groups)

    target_test_size = 0.20 * len(df)
    test_indices = []
    train_indices = []

    for group in scaffold_groups:
        if len(test_indices) < target_test_size:
            test_indices.extend(group)
        else:
            train_indices.extend(group)

    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()

    print("Training molecules:", len(train_df))
    print("Testing molecules:", len(test_df))

    # Verify that the saved prediction report matches this exact test set.
    predictions = pd.read_csv(PREDICTIONS_PATH)

    expected_smiles = set(test_df[SMILES_COLUMN])
    saved_smiles = set(predictions[SMILES_COLUMN])

    if expected_smiles != saved_smiles:
        raise ValueError(
            "The saved predictions do not match the recreated test split. "
            "Stop here; do not interpret the coverage results."
        )

    # Calculate the six descriptors and fingerprints for training molecules.
    print("\nPreparing training molecules...")

    train_smiles = train_df[SMILES_COLUMN].tolist()
    train_targets = train_df[TARGET].to_numpy(dtype=float)

    train_fingerprints = [
        get_fingerprint(smiles) for smiles in train_smiles
    ]

    train_descriptors = np.array([
        calculate_features(smiles)[0][:6]
        for smiles in train_smiles
    ])

    descriptor_min = train_descriptors.min(axis=0)
    descriptor_max = train_descriptors.max(axis=0)

    # Analyze all test molecules, starting with the largest prediction errors.
    predictions = predictions.sort_values(
        "absolute_error", ascending=False
    ).reset_index(drop=True)

    output_rows = []

    print("\nComparing test molecules against training molecules...")

    for _, row in predictions.iterrows():
        smiles = row[SMILES_COLUMN]

        test_fingerprint = get_fingerprint(smiles)

        similarities = DataStructs.BulkTanimotoSimilarity(
            test_fingerprint,
            train_fingerprints,
        )

        nearest_index = int(np.argmax(similarities))
        nearest_similarity = float(similarities[nearest_index])

        nearest_smiles = train_smiles[nearest_index]
        nearest_target = float(train_targets[nearest_index])

        descriptors = np.array(get_descriptors(smiles), dtype=float)

        result = {
            "smiles": smiles,
            "actual_log_solubility": row["actual_log_solubility"],
            "predicted_log_solubility": row["predicted_log_solubility"],
            "absolute_error": row["absolute_error"],
            "nearest_training_smiles": nearest_smiles,
            "nearest_training_solubility": nearest_target,
            "nearest_tanimoto_similarity": nearest_similarity,
            "target_difference_from_nearest_training_molecule": (
                float(row["actual_log_solubility"]) - nearest_target
            ),
        }

        # Check whether each descriptor lies outside its training range.
        outside_descriptors = []

        for i, name in enumerate(DESCRIPTOR_NAMES):
            result[name] = descriptors[i]
            result[f"{name}_outside_training_range"] = bool(
                descriptors[i] < descriptor_min[i]
                or descriptors[i] > descriptor_max[i]
            )

            if result[f"{name}_outside_training_range"]:
                outside_descriptors.append(name)

        result["descriptors_outside_training_range"] = (
            ", ".join(outside_descriptors) if outside_descriptors else "None"
        )

        output_rows.append(result)

    results = pd.DataFrame(output_rows)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\n===== TRAINING COVERAGE: 10 LARGEST ERRORS =====")

    display_columns = [
        "smiles",
        "absolute_error",
        "nearest_tanimoto_similarity",
        "nearest_training_solubility",
        "descriptors_outside_training_range",
    ]

    print(
        results[display_columns]
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    print("\n===== SIMILARITY SUMMARY =====")
    print(
        "Mean nearest-training similarity:",
        round(results["nearest_tanimoto_similarity"].mean(), 3),
    )
    print(
        "Median nearest-training similarity:",
        round(results["nearest_tanimoto_similarity"].median(), 3),
    )
    print(
        "Largest-error molecules with similarity below 0.30:",
        int(
            (
                (results["nearest_tanimoto_similarity"] < 0.30)
                & (results.index < 10)
            ).sum()
        ),
        "out of 10",
    )

    print("\nSaved detailed report:", OUTPUT_PATH)
    print("Analyzed test molecules:", len(results))
    print("Saved model and Streamlit app were not modified.")


if __name__ == "__main__":
    main()
