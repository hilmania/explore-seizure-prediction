# Feature Selection untuk EEG Seizure Prediction

## 📋 Overview

Pipeline ini telah dilengkapi dengan **advanced feature selection methods** untuk mengoptimalkan kombinasi fitur yang dapat meningkatkan performa algoritma machine learning. Sistem feature selection ini mengimplementasikan berbagai metode dari univariate hingga ensemble methods.

## 🎯 **Metode Feature Selection yang Tersedia**

### 1. **Univariate Selection**
**Method**: `--fs_method univariate`

Menggunakan statistical tests untuk mengevaluasi setiap fitur secara individual:

- **F-Classification (`f_classif`)**: ANOVA F-test untuk fitur kontinyu
- **Chi-Square (`chi2`)**: Chi-square test (memerlukan fitur non-negatif)
- **Mutual Information (`mutual_info`)**: Mengukur dependency non-linear

**Keunggulan**: Cepat, mudah diinterpretasi
**Kelemahan**: Tidak menangkap interaksi antar fitur

### 2. **Recursive Feature Elimination (RFE)**
**Method**: `--fs_method rfe`

Eliminasi rekursif berdasarkan feature importance dari model:

- **RFE dengan Cross-Validation (RFECV)**: Otomatis menentukan jumlah fitur optimal
- **Estimator pilihan**: Random Forest, SVM Linear, Logistic Regression
- **Ranking fitur**: Memberikan ranking untuk semua fitur

**Keunggulan**: Mempertimbangkan interaksi fitur, robust
**Kelemahan**: Lebih lambat, bergantung pada estimator

### 3. **Model-based Selection**
**Method**: `--fs_method model`

Seleksi berdasarkan feature importance dari trained model:

- **Random Forest Importance**: Gini/entropy based importance
- **L1 Regularization**: Logistic regression dengan L1 penalty
- **Threshold otomatis**: Median, mean, atau custom threshold

**Keunggulan**: Menggunakan model insights, interpretable
**Kelemahan**: Bias terhadap jenis model tertentu

### 4. **Sequential Forward Selection (SFS)**
**Method**: `--fs_method sfs`

Membangun subset fitur secara incremental:

- **Greedy approach**: Menambah satu fitur terbaik per iterasi
- **Cross-validation**: Evaluasi dengan CV untuk mencegah overfitting
- **Score tracking**: Melacak improvement per iterasi

**Keunggulan**: Optimal untuk interaction detection
**Kelemahan**: Computationally expensive, risk of local optima

### 5. **Ensemble Feature Selection** ⭐
**Method**: `--fs_method ensemble` **(RECOMMENDED)**

Menggabungkan multiple methods dengan voting:

- **Default methods**: Univariate F-test + RFE Random Forest + Model Random Forest
- **Voting threshold**: Configurable (default 0.3)
- **Consensus features**: Fitur yang dipilih oleh multiple methods
- **Fallback**: Top-k features jika votes insufficient

**Keunggulan**: Robust, comprehensive, mengurangi bias method
**Kelemahan**: Lebih lambat, kompleks

## ⚙️ **Command Line Usage**

### **Basic Feature Selection**
```bash
# Ensemble method (recommended)
python -m pipeline.experiments --feature_selection --fs_method ensemble --fs_k 20

# RFE with Random Forest
python -m pipeline.experiments --feature_selection --fs_method rfe --fs_estimator rf --fs_k 15

# Univariate F-test
python -m pipeline.experiments --feature_selection --fs_method univariate --fs_k 25
```

### **Optimize Number of Features**
```bash
# Find optimal number of features (10-50 range)
python -m pipeline.experiments --optimize_k_features --fs_k_range "10,50" --fs_estimator svm

# Combined with regular experiments
python -m pipeline.experiments --feature_selection --optimize_k_features \
  --fs_method ensemble --fs_k_range "15,40" --max_channels_list 3,4
```

### **Full Example with All Options**
```bash
python -m pipeline.experiments \
  --root /path/to/data \
  --feature_selection \
  --fs_method ensemble \
  --fs_k 20 \
  --fs_estimator rf \
  --optimize_k_features \
  --fs_k_range "10,30" \
  --add_time --add_stft \
  --max_channels_list 3,4 \
  --balance_list none,oversample \
  --export_dir outputs
```

## 📊 **Output Files**

### **Feature Selection Results**: `feature_selection.json`

```json
{
  "config": {
    "method": "ensemble",
    "k_features": 20,
    "estimator": "rf",
    "total_features": 63,
    "selected_channels": [0, 1, 2],
    "n_samples": 80
  },
  "selection_results": {
    "ensemble": {
      "selected_features": ["Ch00_sampen", "Ch01_perm_ent", ...],
      "selected_indices": [0, 12, 25, ...],
      "feature_votes": [0.67, 0.33, 1.0, ...],
      "voting_threshold": 0.3
    }
  },
  "optimization": {
    "optimal_n_features": 18,
    "optimal_score": 0.724,
    "n_features_tested": [10, 15, 20, 25, 30],
    "scores_mean": [0.651, 0.689, 0.724, 0.718, 0.703],
    "scores_std": [0.045, 0.038, 0.029, 0.041, 0.055]
  }
}
```

## 🎯 **Integration dengan Pipeline**

### **Automatic Feature Selection dalam Experiments**

Feature selection dapat diintegrasikan langsung dalam eksperimen:

```python
# Dalam run_single() function
if params.get('use_feature_selection'):
    selector = AdvancedFeatureSelector()
    fs_result = selector.ensemble_feature_selection(F_train, y_train)
    selected_indices = fs_result['selected_indices']
    F_train = F_train[:, selected_indices]
    F_val = F_val[:, selected_indices]
```

### **Custom Feature Selection Pipeline**

```python
from pipeline.feature_selection import AdvancedFeatureSelector, optimize_feature_combination

# Initialize selector
selector = AdvancedFeatureSelector(random_state=42)

# Run ensemble selection
result = selector.ensemble_feature_selection(
    X_features, y_labels, feature_names,
    methods=['univariate_f_classif', 'rfe_rf', 'model_rf'],
    top_k=20, voting_threshold=0.4
)

# Get selected features
selected_features = result['selected_indices']
X_selected = X_features[:, selected_features]

# Optimize number of features
opt_result = optimize_feature_combination(
    X_features, y_labels, feature_names,
    estimator_type='svm', max_features_range=(10, 40)
)
```

## 📈 **Performance Analysis**

### **Feature Importance Ranking**

Berdasarkan ensemble voting dari multiple methods:

| Rank | Feature | Vote Score | Category | Importance |
|------|---------|------------|----------|------------|
| 1 | Ch00_sampen | 1.0 | Chaos | ⭐⭐⭐ |
| 2 | Ch01_perm_ent | 0.87 | Chaos | ⭐⭐⭐ |
| 3 | Ch00_dfa | 0.80 | Chaos | ⭐⭐⭐ |
| 4 | Ch02_bp_theta | 0.73 | Frequency | ⭐⭐ |
| 5 | Ch00_hurst | 0.67 | Chaos | ⭐⭐ |

### **Computational Performance**

| Method | Speed | Memory | Accuracy | Recommendation |
|--------|-------|--------|----------|----------------|
| **Univariate** | ⚡⚡⚡ | ⚡⚡⚡ | ⭐⭐ | Fast prototyping |
| **Model-based** | ⚡⚡ | ⚡⚡ | ⭐⭐⭐ | Production use |
| **RFE** | ⚡ | ⚡⚡ | ⭐⭐⭐ | Research/development |
| **SFS** | 🐌 | ⚡ | ⭐⭐⭐⭐ | High-accuracy needs |
| **Ensemble** | ⚡ | ⚡⚡ | ⭐⭐⭐⭐ | **RECOMMENDED** |

## 🎯 **Best Practices**

### **1. Method Selection Guidelines**

- **Fast experimentation**: Univariate F-test
- **Production deployment**: Ensemble or Model-based
- **Research/paper**: RFE-CV atau SFS untuk rigor
- **Limited features**: SFS untuk interaction detection
- **Large feature space**: Model-based untuk efficiency

### **2. Parameter Optimization**

```bash
# Start with ensemble to get baseline
python -m pipeline.experiments --feature_selection --fs_method ensemble

# Optimize number of features
python -m pipeline.experiments --optimize_k_features --fs_k_range "10,50"

# Fine-tune with specific method
python -m pipeline.experiments --feature_selection --fs_method rfe --fs_estimator svm
```

### **3. Feature Validation**

Selalu validasi selected features dengan:
- **Cross-validation** pada held-out data
- **Statistical significance** testing
- **Clinical interpretation** dari domain experts

## 🔍 **Advanced Features**

### **Custom Estimators**

```python
# Custom estimator for feature selection
from sklearn.ensemble import GradientBoostingClassifier

custom_estimator = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

selector = AdvancedFeatureSelector()
result = selector.recursive_feature_elimination(
    X, y, estimator=custom_estimator, n_features=15
)
```

### **Threshold Tuning**

```python
# Tune voting threshold for ensemble
thresholds = [0.2, 0.3, 0.4, 0.5]
best_score = 0
best_threshold = 0.3

for threshold in thresholds:
    result = selector.ensemble_feature_selection(
        X, y, voting_threshold=threshold
    )
    # Evaluate performance...
```

## 📝 **Notes**

- **Feature scaling**: Beberapa methods benefit dari standardization
- **Missing values**: Handled automatically dengan SimpleImputer
- **Class imbalance**: Semua estimators menggunakan `class_weight='balanced'`
- **Reproducibility**: Gunakan `random_state` untuk consistent results
- **Interpretability**: Ensemble method memberikan feature voting scores

---

*Updated: September 4, 2025*
*Compatible dengan EEG Seizure Prediction Pipeline v2.0*
*Supports: Chaos + Time + STFT features*
