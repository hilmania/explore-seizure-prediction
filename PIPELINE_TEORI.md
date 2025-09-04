# Teori dan Alur Pipeline Prediksi Kejang (Preictal vs Interictal)

Dokumen ini menjelaskan konsep dan alur kerja pipeline yang dibangun untuk prediksi kejang epilepsi pada dataset TUSZ-modified (label: 1=preictal, 0=interictal/normal). Implementasi kodenya berada di paket `pipeline/`.

## Ringkasan Alur
1. Preprocessing (filtering, notch, opsional ICA)
2. Seleksi kanal (wrapper, Sequential Forward Selection/SFS)
3. Ekstraksi fitur (chaos + band power)
4. Pelatihan dan evaluasi beberapa model ML
5. Mekanisme alarm berdasarkan probabilitas preictal per-epoch
6. Evaluasi metrik: ROC-AUC, Accuracy, Specificity, Sensitivity, Precision, FPR, F1, F-beta

Dataset diasumsikan berbentuk array:
- `X`: (n_epoch, n_channel=16, n_sample=1280)
- `y`: (n_epoch,) dengan isi {0,1}
- Default sampling rate `sfreq=256 Hz` → 5 detik per epoch (dapat diubah di `pipeline/config.py`).

---

## 1) Preprocessing
File: `pipeline/preprocess.py`

Tujuan: meningkatkan rasio signal-to-noise dan mengurangi artefak agar fitur lebih stabil.

- Band-pass filter 0.5–40 Hz
  - Hilangkan drift DC dan fokus pada pita yang relevan untuk EEG klinis.
  - Implementasi: IIR Butterworth orde 4 + `filtfilt` (zero-phase) untuk menghindari phase distortion.
- Notch 60 Hz (opsional)
  - Menekan gangguan listrik 60 Hz.
- ICA (opsional, per-epoch)
  - Independent Component Analysis (MNE-ICA) memisahkan komponen sumber independen.
  - Heuristik: komponen dengan kurtosis |k|>5 di-exclude → mitigasi artefak (mis. kedipan, noise tajam).
  - Jika ICA gagal/bermasalah pada epoch tertentu, fallback tanpa ICA.

Catatan: ICA per-epoch bisa lambat; dapat dimatikan dengan `--no_ica` bila eksplorasi cepat.

---

## 2) Seleksi Kanal (Wrapper SFS)
File: `pipeline/channel_select.py`

Tujuan: memilih subset kanal paling informatif untuk klasifikasi, mengurangi dimensi & potensi overfitting.

Metode: Sequential Forward Selection (SFS) berbasis wrapper.
- Ide: mulai dari himpunan kosong, coba tambah satu kanal terbaik di setiap langkah dengan mengevaluasi performa model pada validasi silang (CV).
- Skor default: `roc_auc`.
- Untuk efisiensi seleksi, digunakan fitur ringkas per-kanal (bukan fitur chaos lengkap) saat proses SFS:
  - RMS dan band power kasar (delta/theta/alpha/beta) via Welch.
  - Ini mempercepat iterasi SFS namun tetap menangkap informasi spektral dasar.
- Setelah kanal terpilih (mis. K=6/8), barulah diekstraksi fitur chaos lengkap pada kanal tersebut.

Keluaran: indeks kanal terpilih dan, bila disediakan, nama kanalnya.

---

## 3) Ekstraksi Fitur (Chaos + Band Power)
File: `pipeline/features.py`

Tujuan: memetakan sinyal EEG per-kanal menjadi vektor fitur yang mewakili kompleksitas (chaos) dan karakteristik spektral.

Fitur per kanal:
- Sample Entropy (SampEn)
  - Mengukur kompleksitas/ketidakteraturan sinyal. Lebih tinggi → lebih kompleks.
  - Parameter: `m` (embedding) dan `r` (toleransi, proporsional terhadap std sinyal).
- Permutation Entropy (PermEnt)
  - Entropi berdasarkan pola urutan lokal; robust terhadap noise.
- SVD Entropy
  - Entropi berbasis spektrum singular value; merefleksikan keragaman struktur sinyal.
- Fractal Dimensions
  - Higuchi FD dan Petrosian FD: ukuran fraktalitas/kompleksitas geometrik sinyal.
- DFA (Detrended Fluctuation Analysis)
  - Mengukur korelasi jangka panjang (long-range correlation) pada sinyal non-stasioner.
- Hurst (Hurst exponent)
  - Menggambarkan persistensi: H>0.5 (persisten), H≈0.5 (acak), H<0.5 (anti-persisten).
- Band Powers (delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30 Hz)
  - Energi per pita frekuensi dari Welch.

Penanganan NaN:
- Feature extraction menggunakan `_safe(...)` untuk menangkap error → isi NaN.
- Di tahap model, setiap pipeline model mengandung `SimpleImputer(strategy='median')` agar NaN tidak mematahkan training/evaluasi.

Vektor fitur akhir = konkatenasi fitur tiap kanal terpilih.

---

## 4) Model Machine Learning
File: `pipeline/models.py`

Model yang dibandingkan:
- Logistic Regression (dengan StandardScaler)
- SVM RBF (dengan StandardScaler)
- Random Forest
- Gradient Boosting
- KNN (dengan StandardScaler)

Catatan:
- `class_weight='balanced'` (saat applicable) digunakan untuk mengompensasi ketidakseimbangan kelas.
- Semua pipeline mencakup `SimpleImputer` untuk robust terhadap NaN pada feature matrix.

Output per model pada set eval:
- Probabilitas (atau skor yang dinormalisasi 0–1 bila tidak ada `predict_proba`) → `y_prob`
- Prediksi biner via threshold → `y_pred`

---

## 5) Mekanisme Alarm (Dengan Histeresis dan Refractory)
File: `pipeline/run_pipeline.py` → fungsi `alarm_from_probs(...)`

Tujuan: menandai deteksi preictal secara stabil, mengurangi false alarm sesaat.

Parameter inti:
- `threshold` (warisan): ambang dasar (0–1). Bila `t_on/t_off` tidak diberikan, keduanya = `threshold`.
- Histeresis:
  - `t_on`: ambang untuk menyalakan alarm.
  - `t_off`: ambang untuk mematikan alarm.
  - `min_consecutive_on`: jumlah epoch berurutan ≥ `t_on` agar ON.
  - `min_consecutive_off`: jumlah epoch berurutan < `t_off` agar OFF.
- Refractory:
  - `refractory`: setelah OFF, sistem menunggu N epoch (cooldown) sebelum boleh menyala lagi.

Logika ringkas:
- Keadaan ON/OFF.
- Saat OFF:
  - Jika cooldown>0 → kurangi satu setiap epoch, tidak bisa ON.
  - Jika cooldown==0 dan probabilitas p[i] ≥ `t_on` berturut-turut sebanyak `min_consecutive_on` → ON.
- Saat ON:
  - `alarms[i]=1`.
  - Jika p[i] < `t_off` berturut-turut sebanyak `min_consecutive_off` → OFF, set `cooldown=refractory`.

Contoh:
- `t_on=0.6`, `t_off=0.4`, `min_consecutive_on=2`, `min_consecutive_off=2`, `refractory=3`
- Prob: [0.10, 0.70, 0.65, 0.50, 0.62, 0.61, 0.63, 0.30, 0.70, 0.80]
- Alarm: menyala setelah dua epoch ≥0.6 beruntun; padam setelah dua epoch <0.4; setelah padam, menunggu 3 epoch sebelum bisa menyala lagi.

Pengembangan opsional (dapat ditambahkan):
- Window voting: nyalakan jika ≥M dari N epoch terakhir di atas `t_on`.
- Smoothing: EMA/median filter pada `y_prob` sebelum thresholding.
- Refractory dinamis: lebih panjang setelah alarm panjang.

---

## 6) Metrik Evaluasi
File: `pipeline/metrics.py`

- ROC-AUC
  - Luas di bawah kurva ROC; tidak bergantung threshold; semakin besar semakin baik.
- Accuracy = (TP+TN)/(TP+TN+FP+FN)
- Precision = TP/(TP+FP)
- Sensitivity (Recall) = TP/(TP+FN)
- Specificity = TN/(TN+FP)
- FPR = FP/(FP+TN) = 1−Specificity
- F1 = harmonic mean(Precision, Recall)
- F-beta
  - Menurut bobot beta: Fβ = (1+β²)·(P·R)/(β²·P + R)
  - β>1 menekankan Recall (sensitivitas); cocok untuk deteksi dini preictal.

Catatan: pemilihan threshold memengaruhi metrik berbasis keputusan (Accuracy, Precision, Recall, F1, Fβ, Specificity, FPR) namun tidak memengaruhi ROC-AUC.

---

## 7) I/O dan Reproducibility
- Input data: `.npy` berpasangan `*_X.npy` dan `*_y.npy` (struktur folder train/eval)
- Output:
  - Model: `outputs/model_<nama>.joblib`
  - Prediksi: `outputs/<model>_y_prob.npy`, `outputs/<model>_y_pred.npy`
  - Alarm model terbaik: `outputs/<best>_alarms.npy`
  - Seleksi kanal: `outputs/selected_channels.json` (indeks, nama terpilih, dan seluruh daftar nama)
  - Metrik ringkas: `outputs/metrics.json` (termasuk meta: threshold, histeresis, refractory, dll) dan `outputs/metrics.csv`

---

## 8) Tips Praktis
- ICA lambat → pakai `--no_ica` saat eksperimen cepat; aktifkan saat final.
- Tuning threshold/histeresis/refractory untuk trade-off sensitivitas vs false alarm.
- Pertimbangkan split per-subjek untuk menilai generalisasi antar pasien.
- Tuning model (RF trees, SVM C/γ, jumlah kanal terpilih) untuk peningkatan kinerja.

---

## 9) Referensi Singkat Konsep
- Entropy (Sample/Permutation/SVD): ukuran kompleksitas dan keteraturan sinyal.
- Fractal Dimension (Higuchi/Petrosian): karakterisasi bentuk sinyal multi-skala.
- DFA & Hurst: korelasi jangka panjang pada proses non-stasioner.
- Wrapper SFS: pemilihan fitur/kanal berbasis performa model (bukan hanya statistik univariat).
- Histeresis & Refractory: stabilisasi keputusan deteksi berbasis probabilitas waktu-ke-waktu.

---

Untuk diskusi lanjutan, kita bisa menambahkan mode voting jendela, smoothing probabilitas, dan analisis latensi alarm relatif ke onset preictal untuk kebutuhan prediksi dini.
