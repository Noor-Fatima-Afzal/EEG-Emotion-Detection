# EEG Emotion Detection

A lightweight Streamlit app for **EEG emotion inference** from **EDF** files.  
It performs preprocessing (notch/bandpass, re-reference), per-epoch feature extraction (bandpowers, Hjorth, ratios, differential entropy), optional **scikit-learn** model inference, and rich **signal visualizations** (raw trace, spectrogram, PSD, bandpower bars, topographic maps).

> Works even without a trained model via a small heuristic fallback.

---

## ✨ Features

- **EDF ingestion** (MNE) with Windows-friendly temp file handling
- **Preprocessing**: notch (50/60 Hz), bandpass (low/high), average re-reference
- **Epoching** with length/overlap controls and optional resampling
- **Features per epoch**:
  - Band powers: δ(1–4), θ(4–8), α(8–13), β(13–30), γ(30–45)
  - Ratios: α/β, β/(α+θ)
  - Hjorth: activity, mobility, complexity
  - Differential entropy (per band)
  - Aggregate stats across channels (mean/std/median)
- **Visualizations**:
  - Channel explorer (raw trace for a selected channel & time window)
  - Spectrogram (time–frequency heatmap)
  - PSD (Welch) for multiple channels + mean
  - Bandpower bar chart per channel
  - Topographic maps (alpha/theta/beta) when montage/positions are available
- **Inference**:
  - Optional scikit-learn classifier (`.pkl`) with automatic class alignment
  - Heuristic fallback when no model is provided
- **Outputs**:
  - Per-epoch predictions CSV download
  - Average class probabilities & prediction timeline

---

## 📊 Visualizations

- **Channel Explorer**  
  Select a channel and a time window (start time + preview seconds) to inspect the raw trace.

- **Spectrogram**  
  Time–frequency heatmap (1–45 Hz by default) using `scipy.signal.spectrogram`.

- **PSD (Welch)**  
  Power spectral density curves for up to 8 channels + mean curve for a quick overview.

- **Bandpower Bars**  
  Grouped bars per frequency band (δ, θ, α, β, γ) across channels for the selected preview window.

- **Topographic Maps (optional)**  
  Theta/Alpha/Beta bandpower maps plotted over the scalp if your EDF includes electrode positions or a recognizable montage.

---

## 🧪 Feature Extraction

Per epoch (per window), the app computes:

- **Band powers**: δ(1–4), θ(4–8), α(8–13), β(13–30), γ(30–45) via Welch’s method  
- **Ratios**: α/β, β/(α+θ)  
- **Hjorth parameters**: activity (variance), mobility, complexity  
- **Differential entropy** (using band power as variance proxy)  
- **Aggregate channel stats**: mean, std, median across channels  

These are concatenated into a single feature vector per epoch.

---

## ⚙️ Configuration & Controls

### Sidebar → Settings
- **Epoch length (s)**: window length for features/predictions (default 2.0)  
- **Epoch overlap (s)**: overlap between consecutive windows (default 0.5)  
- **Resample to (Hz)**: downsample to stabilize computations (default 128)  
- **Bandpass low / high (Hz)**: pre-filtering range (default 1–45)  
- **Notch filter**: 50/60 Hz line noise removal  
- **Re-reference**: None or Average reference  

### Sidebar → Visuals
- **Preview window (s)**: window size for the Channel Explorer (default 10s)  
- **Spectrogram window (samples)**: nperseg (e.g., 256)  
- **Topographic maps**: toggle; shown only when montage/positions are available  
