```markdown
# Integrated Applied Metrology Pipeline

An end-to-end pipeline combining physics-informed feature engineering, full-space out-of-distribution anomaly detection, dimensionality reduction, and probabilistic Gaussian Process regression with uncertainty quantification.

## Pipeline Architecture
1. **Raw Signal Generation:** Simulates high-dimensional multivariate sensor signals.
2. **Physics Feature Extraction:** Appends power integrals, spectral band ratios, and harmonic resonance modes.
3. **Full-Space OOD Filtering:** Applies `StandardScaler` and `IsolationForest` across all feature dimensions to prevent anomaly leakage into test evaluations.
4. **Probabilistic Modeling:** Trains an Automatic Relevance Determination (ARD) Gaussian Process Regressor (`RBF` + `WhiteKernel`) on clean inlier data.
5. **Validation Visualization:** Renders PCA 2D projections, 1:1 prediction fit plots ($R^2 \approx 0.998$), and sample-wise $95\%$ confidence intervals.

## Prerequisites
```bash
pip install numpy scikit-learn matplotlib
