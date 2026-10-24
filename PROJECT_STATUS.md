# Project Status

## Stage 0 — Environment and Project Setup

Completed:

- Project directory created
- Python virtual environment created
- Scientific Python libraries installed
- RDKit installed
- Git repository initialized
- Project folder structure created
- Documentation files created

---

## Stage 1 — Machine Learning Foundations

### Completed

#### Mathematics and Linear Regression
- Variables and predictions
- Model parameters
- Bias and weights
- Mean Squared Error (MSE)
- Derivative and gradient
- Gradient descent
- Linear regression from scratch
- Multiple data points
- Multiple features
- Matrix-based prediction
- Dot products

#### Generalization
- Train/test split
- Training error vs test error
- Validation dataset
- Training / validation / test roles
- Generalization

#### Model Complexity
- Polynomial features
- Underfitting
- Good fit
- Overfitting
- Model complexity

#### Feature Scaling
- Mean and standard deviation
- Standardization
- Why feature scaling matters
- Gradient descent with unscaled features
- Numerical instability
- Gradient descent with scaled features

#### Model Evaluation
- MAE
- MSE
- RMSE
- R²
- Comparing multiple models

### Important Lessons

- Training data is used to learn model parameters.
- Validation data is used for model selection and hyperparameter tuning.
- Test data should remain unseen until final evaluation.
- Training performance alone does not tell us whether a model generalizes.
- More complex models can perform worse on unseen data.
- Feature scaling can make gradient descent much more stable.
- MAE measures average absolute error.
- MSE penalizes large errors more strongly.
- RMSE expresses error in the original target units.
- R² measures how much variation in the target is explained by the model.

### Final Evaluation Example

Model A:

- MAE ≈ 0.14
- MSE ≈ 0.022
- RMSE ≈ 0.148
- R² ≈ 0.997

Model B:

- MAE = 1.2
- MSE = 1.6
- RMSE ≈ 1.265
- R² = 0.8

Model A generalized much better on this example.

---

## Current Stage

Stage 1 is complete.

## Next Stage

### Stage 2 — First Real Molecular Dataset

Planned:

1. Understand molecular data
2. Understand molecular representations
3. Load a real molecular dataset
4. Use RDKit to represent molecules
5. Extract molecular descriptors
6. Create feature matrices
7. Define a molecular property prediction problem
8. Split molecular data into train/validation/test sets
9. Train a regression model
10. Evaluate molecular predictions
11. Visualize molecules and predictions

### Learning Rule

For every major ML concept:

**Understand → derive → implement from scratch → test → use libraries → apply to molecules.**