# MolecularAI - Model Card



## 1. Model Overview



- **Project:** MolecularAI

- **Task:** Aqueous solubility prediction using the ESOL dataset

- **Target:** Measured log solubility in moles per litre

- **Algorithm:** RandomForestRegressor

- **Primary use:** Educational research and exploratory molecular property prediction



## 2. Training Data



- Original dataset rows: 1,128

- Cleaned unique canonical molecular structures: 1,117

- Duplicate structures: Aggregated using mean measured solubility

- Invalid SMILES: Excluded during cleaning



## 3. Input Features



The model uses 2,054 features:



- Six molecular descriptors: molecular weight, hydrogen-bond donors, hydrogen-bond acceptors, rotatable bonds, topological polar surface area (TPSA), and LogP

- Morgan fingerprints: radius 2, 2,048 bits



## 4. Model Configuration



- Estimator: RandomForestRegressor

- Trees: 200

- Maximum depth: 12

- Minimum samples per leaf: 2

- Random state: 42



## 5. Evaluation Status



Exploratory scaffold-split experiments compared alternative Random Forest settings. On one development split, the leaf-size-2 configuration achieved:



- MAE: 0.5920

- RMSE: 0.7447

- R-squared: 0.7688



These metrics belong to a separately trained evaluation model, not a direct evaluation of the saved candidate model. The development splits have been used during model selection and are not an untouched final benchmark.



The saved candidate was subsequently retrained on all 1,117 cleaned molecules. Its training performance is not independent test performance. External validation has not been performed.



## 6. Limitations



- Predictions are estimates, not experimental measurements.

- Accuracy may vary across chemical families and outside the training data's chemical space.

- Similarity to known training molecules does not guarantee reliable predictions.

- Experimental testing and independent external validation are needed before practical decisions rely on predictions.



## 7. Responsible Use



Use this model for learning, exploratory analysis, and hypothesis generation. Do not treat its output alone as evidence of drug efficacy, safety, or suitability for human use.
