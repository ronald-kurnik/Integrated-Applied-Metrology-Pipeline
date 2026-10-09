import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# ==========================================
# 1. SYNTHETIC SENSOR DATA GENERATION
# ==========================================
num_samples = 160
num_sensor_features = 20

# Raw sensor measurements
raw_sensor_matrix = np.random.normal(loc=2.0, scale=0.8, size=(num_samples, num_sensor_features))

# Underlying physical property (film thickness nm)
true_thickness = (
    15.0 
    + 3.2 * np.sin(raw_sensor_matrix[:, 0]) 
    + 1.8 * (raw_sensor_matrix[:, 5] ** 2) 
    + np.random.normal(0, 0.15, size=num_samples)
)

# Inject out-of-distribution / defective process sample shift
raw_sensor_matrix[145:, :] += np.random.uniform(4.0, 7.0, size=(15, num_sensor_features))

# ==========================================
# 2. PHYSICS-INFORMED FEATURE ENGINEERING
# ==========================================
def extract_physics_features(X_raw: np.ndarray) -> np.ndarray:
    total_power = np.sum(X_raw**2, axis=1, keepdims=True)
    band_ratio = (np.mean(X_raw[:, :5], axis=1) / (np.mean(X_raw[:, 15:], axis=1) + 1e-8)).reshape(-1, 1)
    
    sin_mode = np.sin(X_raw[:, 0:1])
    squared_mode = (X_raw[:, 5:6] ** 2)
    
    return np.hstack([X_raw, total_power, band_ratio, sin_mode, squared_mode])

X_phys = extract_physics_features(raw_sensor_matrix)

# ==========================================
# 3. FEATURE SCALING & FULL-SPACE OOD DETECTION
# ==========================================
scaler = StandardScaler()
X_phys_scaled = scaler.fit_transform(X_phys)

# FIX: Fit IsolationForest on FULL feature space to catch high-dimensional anomalies
iso_forest = IsolationForest(contamination=0.10, random_state=42)
inlier_mask = iso_forest.fit_predict(X_phys_scaled) == 1

X_clean = X_phys_scaled[inlier_mask]
y_clean = true_thickness[inlier_mask]

# PCA strictly for 2D visualization
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_phys_scaled)
X_clean_2d = X_pca_2d[inlier_mask]
X_anomalies_2d = X_pca_2d[~inlier_mask]

# ==========================================
# 4. TRAIN / TEST SPLIT & PROBABILISTIC REGRESSION
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.25, random_state=42
)

kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=np.ones(X_train.shape[1]), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e1))

gp_model = gp.GaussianProcessRegressor(
    kernel=kernel, 
    normalize_y=True, 
    n_restarts_optimizer=10, 
    random_state=42
)
gp_model.fit(X_train, y_train)

y_pred_mean, y_pred_std = gp_model.predict(X_test, return_std=True)

r2 = r2_score(y_test, y_pred_mean)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))

# ==========================================
# 5. VISUALIZATION
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: PCA Latent Space & Out-of-Distribution Detection
axes[0].scatter(X_clean_2d[:, 0], X_clean_2d[:, 1], c='#1f77b4', label='In-Spec Material Data', alpha=0.7)
axes[0].scatter(X_anomalies_2d[:, 0], X_anomalies_2d[:, 1], c='crimson', marker='x', s=70, linewidths=2, label='Detected Anomalies (OOD)')
axes[0].set_title("1. PCA Latent Space & OOD Detection", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(loc='upper right')

# Plot 2: Probabilistic Regression Performance
axes[1].scatter(y_test, y_pred_mean, color='#2ca02c', edgecolors='k', alpha=0.85, s=45, label='Test Predictions')
axes[1].plot([y_clean.min(), y_clean.max()], [y_clean.min(), y_clean.max()], 'k--', lw=1.5, label='Ideal Fit (1:1)')
axes[1].set_title(f"2. Probabilistic Regression ($R^2 = {r2:.3f}$, RMSE = {rmse:.3f})", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Actual Material Property (Thickness nm)")
axes[1].set_ylabel("Predicted Material Property (Thickness nm)")
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(loc='upper left')

# Plot 3: Prediction Intervals & Uncertainty Quantification
sample_indices = np.arange(len(y_test))
axes[2].errorbar(sample_indices, y_pred_mean, yerr=1.96 * y_pred_std, fmt='o', color='#1f77b4', 
                 ecolor='orange', elinewidth=1.8, capsize=4, label='Predicted Mean ± 95% UQ')
axes[2].scatter(sample_indices, y_test, color='black', zorder=5, marker='s', s=30, label='Ground Truth Value')
axes[2].set_title("3. Prediction Intervals & Uncertainty Quantification", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Test Sample Index")
axes[2].set_ylabel("Material Property (nm)")
axes[2].grid(True, linestyle='--', alpha=0.5)
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.show()