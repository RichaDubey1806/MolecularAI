
from pathlib import Path
import json
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

candidate_path = (
    PROJECT_ROOT / "models" / "solubility_rf_clean_candidate.joblib"
)
candidate_config_path = (
    PROJECT_ROOT / "models" / "solubility_rf_clean_candidate_config.json"
)
original_path = (
    PROJECT_ROOT / "models" / "solubility_rf_baseline.joblib"
)

print("=" * 55)
print("MOLECULARAI CANDIDATE MODEL VERIFICATION")
print("=" * 55)

for path in [candidate_path, candidate_config_path, original_path]:
    print(f"\n{path.name}")
    print(f"Exists: {path.exists()}")

    if path.exists():
        print(f"Size: {path.stat().st_size:,} bytes")

if not candidate_path.exists():
    raise FileNotFoundError("Candidate model file is missing.")

if not candidate_config_path.exists():
    raise FileNotFoundError("Candidate config file is missing.")

candidate = joblib.load(candidate_path)

with open(candidate_config_path, "r", encoding="utf-8") as file:
    config = json.load(file)

print("\n--- Candidate Model Details ---")
print("Model type:", type(candidate).__name__)
print("Number of trees:", candidate.n_estimators)
print("Maximum depth:", candidate.max_depth)
print("Minimum samples per leaf:", candidate.min_samples_leaf)
print("Features expected:", candidate.n_features_in_)
print("Training molecules in config:", config["training_samples"])
print("Features in config:", config["total_features"])

assert candidate.n_estimators == 200
assert candidate.n_features_in_ == 2054
assert config["training_samples"] == 1117
assert config["total_features"] == 2054

print("\nPASS: Candidate model and configuration look consistent.")
print("The original model has not been changed by this verification.")
