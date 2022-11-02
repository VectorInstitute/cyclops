# Drift Detection

Dataset drift (aka dataset shift) occurs when the underlying distribution of the source data used to build a model differs from the distribution of the target data used to test the model during deployment. When the difference between the joint probability distribution of the source and target data is sufficient to deteriorate the model’s performance, a shift is considered malignant. A malignant shift could occur for a variety of reasons, for instance sociodemographic inequities and poor resource management. In order to prevent these malignant shifts from occurring, it is important to first characterize them and subsequently understand the circumstances under which they arise.

We use multivariate tests to detect malignant shifts in the joint distribution between source and target data. Our tool performs: 1) dimensionality reduction to obtain a latent representation of the source and target data and 2) statistical testing to identify if a malignant shift has occurred between the latent representation of the source and target data 3) uses Shapley additive explanations (SHAP) values to obtain explanations of the features that drive patient-specific predictions.

## Reductor

- NoRed
- PCA
- SRP
- kPCA
- Isomap
- BBSDs_FFNN
- BBSDh_FFNN
- BBSDs_CNN
- BBSDh_CNN
- BBSDs_LSTM
- BBSDh_LSTM

## Tester

- MMD
- LSDD
- LK
- Classifier
- Context-Aware MMD

## Explainer

- Shap values

## Experiments

### Clinical Experiments

We will use prior knowledge to evaluate real-life scenarios that may cause malignant shifts in healthcare data. Evaluating real-life settings that cause malignant shift can reveal sociodemographic factors that result in biases of machine learning models (e.g. demographic, gender, income, education, employment status), problems affecting care by providers (e.g. overworked, not compensated, lack of resources), and inefficiencies in the management of care that lead to poor outcomes (e.g. resource allocation, staffing). The types of experiments one can run include, but are not limited to:

- Time (e.g. Covid)
- Seasonal (e.g. summer vs. winter)
- Hospital Type (e.g. academic hospitals vs. community hospitals)
- Age (e.g. paediatric vs. adult)
- Gender (e.g. male vs. female)

### Synthetic Experiments

Synthetic experiments will be performed to evaluate under what magnitudes and types (i.e. covariate, label and concept) of shift models undergo malignant shift. The experiments available are as follows:

- Gaussian Noise Shift
- Binary Noise Shift
- Knockout Shift
- Changepoint Shift
- Multiway Feature Association Shift
