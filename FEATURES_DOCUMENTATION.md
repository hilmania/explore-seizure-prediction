# EEG Seizure Prediction Pipeline - Features Documentation

## 📋 Overview

Pipeline ini mengimplementasikan ekstraksi fitur multi-domain untuk prediksi seizure dari data EEG. Sistem fitur dibagi menjadi tiga kategori utama: **Chaos Features**, **Time Domain Features**, dan **STFT Frequency Features**.

## 🧠 1. Chaos Features (Core Features)
*File: `pipeline/features.py`*

Fitur chaos merupakan fitur utama yang selalu diekstrak untuk setiap channel EEG. Fitur ini menangkap karakteristik nonlinear dan kompleksitas dinamis dari sinyal EEG.

### 1.1 Entropy-based Features

| Fitur | Formula/Method | Deskripsi | Parameter |
|-------|----------------|-----------|-----------|
| **`ch{c}_sampen`** | Sample Entropy | Mengukur regularitas dan kompleksitas sinyal | m=2, r=0.2×std |
| **`ch{c}_perm_ent`** | Permutation Entropy | Mengukur kompleksitas berdasarkan pola ordinal | order=3, delay=1 |
| **`ch{c}_svd_ent`** | SVD Entropy | Entropi berdasarkan Singular Value Decomposition | order=3, delay=1 |

### 1.2 Fractal Dimension Features

| Fitur | Method | Deskripsi | Range Nilai |
|-------|--------|-----------|-------------|
| **`ch{c}_higuchi_fd`** | Higuchi Algorithm | Mengukur dimensi fraktal sinyal | 1.0 - 2.0 |
| **`ch{c}_petrosian_fd`** | Petrosian Method | Estimasi dimensi fraktal berbasis crossing | 1.0 - 2.0 |

### 1.3 Chaos Proxies

| Fitur | Method | Deskripsi | Interpretasi |
|-------|--------|-----------|--------------|
| **`ch{c}_dfa`** | Detrended Fluctuation Analysis | Mengukur korelasi jangka panjang | 0.5=random, 1.0=1/f noise |
| **`ch{c}_hurst`** | Hurst Exponent | Mengukur persistensi dan anti-persistensi | 0.5=random, >0.5=persistent |

### 1.4 Band Power Features (Welch PSD)

| Fitur | Frequency Band | Deskripsi | Asosiasi Klinis |
|-------|----------------|-----------|-----------------|
| **`ch{c}_bp_delta`** | 0.5 - 4 Hz | Delta band power | Deep sleep, patologi |
| **`ch{c}_bp_theta`** | 4 - 8 Hz | Theta band power | Drowsiness, meditation |
| **`ch{c}_bp_alpha`** | 8 - 13 Hz | Alpha band power | Relaxed wakefulness |
| **`ch{c}_bp_beta`** | 13 - 30 Hz | Beta band power | Active thinking, alertness |

## 📊 2. Time Domain Statistical Features
*File: `pipeline/extra_features.py`*

Fitur statistik klasik yang menangkap karakteristik amplitudo dan distribusi sinyal EEG.

### 2.1 Statistical Moments

| Fitur | Formula | Deskripsi | Interpretasi |
|-------|---------|-----------|--------------|
| **`ch{c}_mean`** | μ = Σx/n | Mean amplitudo | Baseline drift, DC offset |
| **`ch{c}_std`** | σ = √(Σ(x-μ)²/n) | Standard deviation | Variabilitas sinyal |
| **`ch{c}_skew`** | Skewness | Asimetri distribusi | >0: tail kanan, <0: tail kiri |
| **`ch{c}_kurt`** | Kurtosis | Ketajaman distribusi | >3: leptokurtic, <3: platykurtic |

### 2.2 Amplitude Features

| Fitur | Formula | Deskripsi | Kegunaan |
|-------|---------|-----------|----------|
| **`ch{c}_ptp`** | max(x) - min(x) | Peak-to-peak amplitude | Dynamic range sinyal |
| **`ch{c}_rms`** | √(Σx²/n) | Root Mean Square | Power rata-rata sinyal |

## 🔊 3. STFT Frequency Features
*File: `pipeline/extra_features.py`*

Fitur frekuensi berbasis Short-Time Fourier Transform untuk analisis spektral yang lebih robust.

### 3.1 STFT Band Powers

| Fitur | Band | Method | Keunggulan vs Welch |
|-------|------|--------|---------------------|
| **`ch{c}_stft_bp_delta`** | 0.5 - 4 Hz | STFT + trapz integration | Resolusi waktu-frekuensi lebih baik |
| **`ch{c}_stft_bp_theta`** | 4 - 8 Hz | STFT + trapz integration | Menangkap transien frekuensi |
| **`ch{c}_stft_bp_alpha`** | 8 - 13 Hz | STFT + trapz integration | Adaptive windowing |
| **`ch{c}_stft_bp_beta`** | 13 - 30 Hz | STFT + trapz integration | Resolusi temporal tinggi |

## ⚙️ 4. Konfigurasi dan Parameter

### 4.1 Feature Configuration
*File: `pipeline/config.py`*

```python
@dataclass
class FeatureConfig:
    sample_entropy_m: int = 2           # Pattern length untuk SampEn
    sample_entropy_r: float = 0.2       # Tolerance untuk SampEn (× std)
    permutation_entropy_order: int = 3  # Order untuk PermEn
    permutation_entropy_delay: int = 1  # Delay untuk PermEn
```

### 4.2 STFT Parameters

```python
# STFT Configuration
nperseg = min(256, len(signal))  # Window length
fs = 256.0                       # Sampling frequency
bands = [(0.5,4,'delta'), (4,8,'theta'), (8,13,'alpha'), (13,30,'beta')]
```

## 🎯 5. Penggunaan dalam Pipeline

### 5.1 Command Line Interface

```bash
# Core chaos features saja (11 fitur × n_channels)
python -m pipeline.experiments

# Dengan time domain features (+6 fitur × n_channels)
python -m pipeline.experiments --add_time

# Dengan STFT features (+4 fitur × n_channels)
python -m pipeline.experiments --add_stft

# Semua fitur (21 fitur × n_channels)
python -m pipeline.experiments --add_time --add_stft
```

### 5.2 Feature Extraction Flow

```
Raw EEG Data (n_epochs × n_channels × n_samples)
           ↓
    Preprocessing (filtering, ICA)
           ↓
    Channel Selection (wrapper method)
           ↓
┌─────────────────┬─────────────────┬─────────────────┐
│  Chaos Features │ Time Features   │ STFT Features   │
│  (Core - 11)    │ (Optional - 6)  │ (Optional - 4)  │
└─────────────────┴─────────────────┴─────────────────┘
           ↓
    Feature Matrix (n_epochs × n_features)
           ↓
    Machine Learning Models
```

## 📈 6. Analisis Dimensi Fitur

### 6.1 Perhitungan Total Fitur

| Kategori | Fitur per Channel | Channels | Total |
|----------|------------------|----------|-------|
| **Chaos (Core)** | 11 | 16 | **176** |
| **Time Domain** | 6 | 16 | **96** |
| **STFT** | 4 | 16 | **64** |
| **Total Maksimum** | **21** | **16** | **336** |

### 6.2 Complexity Analysis

- **Computational Complexity**: O(n_epochs × n_channels × n_samples)
- **Memory Complexity**: O(n_epochs × n_features)
- **Recommended Features**: Chaos + Time (17 fitur/channel) untuk balance performance/speed

## 🔬 7. Feature Significance Analysis

### 7.1 Implemented Tests
- **Permutation Importance**: Mengukur drop performa saat fitur di-shuffle
- **Feature Ablation**: Mengukur kontribusi individual fitur
- **Channel Significance**: ROC-AUC per channel vs random labels

### 7.2 Output Files
- **`feature_significance.json`**: Hasil analisis signifikansi fitur dan channel
- **`statistical_validation.json`**: Validasi statistik antar model
- **`experiments_results.csv`**: Hasil eksperimen lengkap

## 📚 8. Dependencies dan Libraries

### 8.1 Core Libraries
```python
import antropy as ant          # Entropy and chaos features
import numpy as np             # Numerical computing
import scipy.signal           # Signal processing (welch, stft)
import scipy.stats            # Statistical functions
```

### 8.2 Feature-specific Dependencies
- **AntroPy**: Sample entropy, permutation entropy, SVD entropy, fractal dimensions
- **SciPy**: DFA, Hurst exponent, Welch PSD, STFT
- **NumPy**: Statistical moments, RMS, peak-to-peak

## 🎯 9. Best Practices

### 9.1 Feature Selection Strategy
1. **Start with core chaos features** (proven effective for EEG)
2. **Add time domain features** if more statistical info needed
3. **Add STFT features** for frequency-domain insights
4. **Use feature significance analysis** to identify most important features

### 9.2 Performance Considerations
- **Core features**: Fast extraction, proven effectiveness
- **Time features**: Very fast, good for baseline
- **STFT features**: Slower but more informative for spectral analysis

### 9.3 Channel-specific Notes
```python
# Format naming convention
ch{channel_index}_{feature_name}

# Example for channel 0:
ch0_sampen, ch0_mean, ch0_stft_bp_alpha
```

---

## 📝 Notes

- **Channel indexing**: 0-based (ch0, ch1, ..., ch15)
- **NaN handling**: Safe wrapper functions prevent crashes
- **Normalization**: Some features auto-normalized (PermEn, SVD entropy)
- **Frequency bands**: Standard clinical EEG bands
- **Chaos theory**: Features designed for nonlinear dynamics analysis

---

*Generated on: September 4, 2025*
*Pipeline Version: Latest*
*Author: EEG Seizure Prediction Team*
