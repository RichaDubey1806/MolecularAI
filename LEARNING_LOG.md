# Learning Log

## 2026-09-30

### Project Started

Started AI Molecular Discovery & Design Platform.

### Current Stage

Stage 0 â€” Project setup.

### Completed

- Created project repository
- Created directory structure
- Created Python virtual environment
- Installed basic scientific Python libraries
- Installed RDKit
- Initialized Git
- Created initial documentation

### ML Concepts Learned

None yet.

### Next Step

Begin mathematical foundations required for machine learning,
starting with vectors, matrices, functions, derivatives, and loss functions.

## Regression and Generalization

Learned:

1. Train/test split
2. Training error vs test error
3. Generalization
4. Polynomial features
5. Overfitting concept
6. Feature scaling

Important observation:

Linear model:
- Training error â‰ˆ 0.029
- Test error â‰ˆ 0.063

Polynomial model:
- Training error â‰ˆ 0.032
- Test error â‰ˆ 0.292

The polynomial model was more complex but generalized worse to unseen data.

Key lesson:

A model should learn the underlying pattern rather than memorize the training data.

# Stage 1 â€” Regression and Model Evaluation

## What I Learned

### Linear Regression

A linear model predicts:

y_hat = Xw + b

where:

- X = input features
- w = learned weights
- b = bias
- y_hat = prediction

### Gradient Descent

The model learns parameters by repeatedly moving them in the direction that reduces the loss.

w_new = w - learning_rate Ã— gradient

### Train / Validation / Test

- Training set â†’ learn parameters
- Validation set â†’ choose models and hyperparameters
- Test set â†’ final evaluation on unseen data

### Feature Scaling

Standardization:

X_scaled = (X - mean) / standard_deviation

After standardization:

- Mean â‰ˆ 0
- Standard deviation â‰ˆ 1

Feature scaling can make gradient descent much more stable.

### Underfitting

The model is too simple to capture the underlying pattern.

### Overfitting

The model learns the training data too specifically and performs worse on unseen data.

### Evaluation Metrics

MAE:

Average absolute prediction error.

MSE:

Average squared prediction error.

RMSE:

Square root of MSE, expressed in the original target units.

RÂ²:

Measures how much variation in the target is explained by the model.

## Important Experiment

Two models were evaluated on the same data.

Model A:

- MAE â‰ˆ 0.14
- MSE â‰ˆ 0.022
- RMSE â‰ˆ 0.148
- RÂ² â‰ˆ 0.997

Model B:

- MAE = 1.2
- MSE = 1.6
- RMSE â‰ˆ 1.265
- RÂ² = 0.8

Model A performed substantially better.

## Key Understanding

A machine learning model should not simply memorize training examples.

The important question is:

> Can the model learn a useful pattern that works on previously unseen data?

This concept will become especially important when predicting properties of molecules that the model has never seen before.

## Stage 1 Completed

The basic mathematical foundation of supervised regression is now complete.

Next:

**Apply these ideas to real molecular data.**

## Molecular Solubility Experiment â€” Model Comparison

**Dataset:** Delaney/ESOL, 1,128 molecules
**Target:** Measured aqueous log-solubility
**Validation:** Five-fold shuffled cross-validation on the training partition

| Model | Mean CV MAE | Mean CV RÂ² |
|---|---:|---:|
| LogP-only linear regression | 0.905 Â± 0.025 | 0.684 Â± 0.048 |
| Six-descriptor linear regression | 0.770 Â± 0.050 | 0.777 Â± 0.020 |
| Six-descriptor Random Forest | 0.511 Â± 0.045 | 0.886 Â± 0.014 |
| Morgan fingerprints Random Forest | 0.931 Â± 0.072 | 0.630 Â± 0.043 |
| Combined descriptors + fingerprints Random Forest | 0.499 Â± 0.045 | 0.893 Â± 0.011 |

### Key observations
- LogP is the strongest feature by built-in Random Forest importance, but it is not sufficient on its own.
- Adding molecular descriptors improves the linear model.
- Random Forest substantially outperforms linear regression with the same six descriptors.
- Combining descriptors and Morgan fingerprints gives a small additional improvement over descriptors alone.
- The linear-model residual plots showed no obvious curved pattern in the inspected views.

### Limitations and next steps
- These results are exploratory because the same test partition has been inspected repeatedly.
- Feature importance is predictive, not proof of causation.
- Random cross-validation does not guarantee generalization to structurally novel molecules.
- Next: evaluate with a molecular scaffold split and compare performance on structurally distinct molecules.
## Molecular Solubility Experiment â€” Scaffold Split

**Dataset:** Delaney/ESOL, 1,128 molecules
**Target:** Measured aqueous log-solubility
**Split:** 80/20 scaffold-based split; no scaffold overlap between training and test sets.

| Model | Test MAE | Test RMSE | Test RÂ² |
|---|---:|---:|---:|
| Six-descriptor Random Forest | 0.753 | 0.997 | 0.760 |
| Morgan fingerprints Random Forest | 1.312 | 1.713 | 0.292 |
| Combined descriptors + fingerprints Random Forest | 0.714 | 0.949 | 0.783 |

### Key observations

- The combined model performed best on this scaffold test set.
- Compared with the random split, test performance decreased. This suggests that predicting molecules from unseen structural scaffolds is more challenging.
- The scaffold test set had a lower average solubility than the training set, so the performance difference may reflect both structural novelty and target-distribution differences.
- Residual analysis showed a tendency to overpredict solubility for poorly soluble molecules.
- Mean residual (actual âˆ’ predicted) was âˆ’0.252, and MAE was 0.714.
- Residual patterns can reveal systematic errors, but they do not by themselves prove a particular physical mechanism.

### Limitations and next steps

- This is one scaffold split, not a complete estimate of performance on all unseen molecular structures.
- The random and scaffold test sets contain different molecules and target distributions, so their scores are not a perfectly controlled comparison.
- The test set has already been inspected; it should not be treated as an untouched final benchmark.
- Next: perform scaffold-aware cross-validation and examine how much model performance varies across different held-out scaffold groups.
### Scaffold-aware cross-validation results

**Method:** Five-fold `GroupKFold`, grouping molecules by Bemisâ€“Murcko scaffold strings. Each scaffold group remains entirely within one fold.

**Combined descriptor + Morgan fingerprint Random Forest:**

| Fold | Validation molecules | MAE | RÂ² |
|---|---:|---:|---:|
| 1 | 317 | 0.581 | 0.795 |
| 2 | 254 | 0.645 | 0.679 |
| 3 | 110 | 0.630 | 0.891 |
| 4 | 110 | 0.775 | 0.747 |
| 5 | 111 | 0.606 | 0.835 |

**Pooled out-of-fold metrics:**
- MAE: 0.632
- RMSE: 0.852
- RÂ²: 0.830

### Interpretation and limitations

- Random 5-fold CV MAE was 0.499, while pooled scaffold CV MAE was 0.632.
- The increase in MAE suggests that prediction becomes more difficult when scaffold groups are held out.
- The folds have very different sizes and target distributions. Two folds are dominated by a single large scaffold group, so results must be interpreted cautiously.
- The pooled metrics summarize out-of-fold predictions across all training molecules; they do not replace an untouched final test set.
- Next: compare descriptor-only and combined models under the same scaffold-aware folds, then examine uncertainty and errors on structurally unfamiliar molecules.
### Descriptor-only vs combined model

Both models were evaluated using the same five scaffold-aware folds.

| Model | Pooled CV MAE | Pooled CV RÂ² |
|---|---:|---:|
| Six-descriptor Random Forest | 0.642 | 0.824 |
| Combined descriptors + Morgan fingerprints Random Forest | 0.632 | 0.830 |

**Observation:** Adding Morgan fingerprints reduced pooled MAE by 0.010 and increased RÂ² by 0.006.

**Interpretation:** Fingerprints provided only a small additional improvement under this scaffold-aware evaluation. The six molecular descriptors already captured substantial predictive information for this dataset.

**Limitation:** The improvement has not been established as statistically significant. Fold imbalance and scaffold-group composition remain important limitations.

**Next:** Examine whether prediction errors increase for molecules that are structurally dissimilar to the training molecules.
### Applicability-domain exploration: fingerprint similarity

**Method:** For each scaffold-CV validation molecule, calculated its maximum Morgan fingerprint Tanimoto similarity to molecules in that fold's training set. Compared maximum similarity with out-of-fold absolute prediction error.

| Maximum similarity | Molecules | Mean similarity | MAE |
|---|---:|---:|---:|
| 0â€“0.2 | 113 | 0.167 | 0.627 |
| 0.2â€“0.4 | 497 | 0.298 | 0.661 |
| 0.4â€“0.6 | 241 | 0.485 | 0.585 |
| 0.6â€“0.8 | 40 | 0.668 | 0.663 |
| 0.8â€“1.0 | 11 | 0.932 | 0.254 |

**Pearson correlation between maximum similarity and absolute error:** âˆ’0.066.

### Interpretation and limitations

- The observed linear association between maximum fingerprint similarity and absolute error was very weak.
- MAE did not decrease consistently as similarity increased.
- The highest-similarity group had only 11 molecules, so its lower MAE is uncertain.
- Fingerprint similarity alone did not reliably explain prediction errors in this experiment.
- Further investigation could consider target range, molecular descriptors, scaffold groups, and prediction uncertainty.
- This analysis is exploratory and does not establish that molecular similarity is unimportant.

**Data integrity check:** All arrays contained 902 entries, and the target array matched the dataframe's target order.

**Next:** Investigate whether the model's errors depend on the actual solubility range and whether prediction uncertainty can flag unreliable predictions.
### First uncertainty experiment â€” Random Forest tree spread

**Method:** Trained a 200-tree Random Forest on the scaffold training set. Calculated the standard deviation of individual tree predictions for each molecule in the scaffold test set.

**Tree-spread summary:**
- Mean: 0.757 logS units
- Minimum: 0.311
- Maximum: 1.590

**Tree spread vs absolute prediction error:**

| Group | Molecules | Mean tree spread | MAE |
|---|---:|---:|---:|
| Low spread | 76 | 0.527 | 0.637 |
| Medium spread | 75 | 0.710 | 0.764 |
| High spread | 75 | 1.038 | 0.742 |

**Pearson correlation:** 0.050.

### Interpretation and limitations

- The overall linear association between tree spread and absolute error was very weak.
- The high-spread group did not have the highest MAE; the medium-spread group did.
- Tree-to-tree disagreement is an uncertainty proxy, not a calibrated prediction interval.
- Random Forest trees are correlated and can agree while still being wrong.
- This was exploratory analysis on a previously inspected scaffold test set, not independent validation.

**Next:** Investigate a more principled uncertainty method, such as conformal prediction, and evaluate its prediction-interval coverage on data not used to construct the intervals.
### Split conformal prediction â€” initial scaffold evaluation

**Model:** Random Forest, 200 trees, `min_samples_leaf=2`
**Target coverage:** 90% (`alpha = 0.10`)
**Calibration set:** 364 molecules across 12 scaffold groups
**Model-training set:** 538 molecules across 47 scaffold groups
**Scaffold overlap between fit and calibration sets:** 0

**Calibration results:**
- Mean absolute error: 0.639
- Median absolute error: 0.512
- Maximum absolute error: 4.699
- Finite-sample conformal threshold: 1.256 logS units

**Scaffold test results:**
- Test molecules: 226
- Covered molecules: 180
- Empirical coverage: 79.6%
- Mean interval width: approximately 2.513 logS units

### Interpretation and limitations

- Observed coverage was below the nominal 90% target.
- The calibration and test sets may differ in molecular and target distributions, which can affect coverage.
- Standard split-conformal coverage guarantees rely on exchangeability; they do not automatically guarantee coverage under scaffold distribution shift.
- This was exploratory evaluation on a previously inspected test set, not an untouched final benchmark.

**Next:** Examine the molecules whose actual values fell outside their prediction intervals and check whether misses are concentrated in particular solubility ranges or scaffold groups.
### Training vs scaffold test target-distribution analysis

Compared measured ESOL solubility (`logS`) between the 902-molecule scaffold training set and the 226-molecule scaffold test set.

**Summary statistics**
- Training mean: âˆ’2.862 logS
- Scaffold test mean: âˆ’3.799 logS
- Training molecules below âˆ’6: 72/902 (7.98%)
- Test molecules below âˆ’6: 33/226 (14.60%)

**Percentage by solubility range**

| Solubility range | Training | Scaffold test |
|---|---:|---:|
| Below âˆ’8 | 1.77% | 4.42% |
| âˆ’8 to âˆ’6 | 6.21% | 10.18% |
| âˆ’6 to âˆ’4 | 19.62% | 26.11% |
| âˆ’4 to âˆ’2 | 35.59% | 42.04% |
| âˆ’2 to 0 | 29.60% | 15.93% |
| Above 0 | 7.21% | 1.33% |

**Interpretation:** The scaffold test set has a lower mean measured solubility, a larger proportion of molecules below âˆ’6, and fewer molecules above zero. This indicates a difference in target distributions between the training and test sets.

The earlier conformal evaluation achieved 79.6% empirical coverage against a nominal 90% target. The distribution shift is a plausible contributor, but this analysis alone cannot determine its causal contribution. Unseen molecular scaffolds may also affect performance. The above-zero test group contains only three molecules, so conclusions about that range are highly uncertain.

**Limitation:** The test set has already been used for exploratory analysis; results should not be presented as an untouched final benchmark.
