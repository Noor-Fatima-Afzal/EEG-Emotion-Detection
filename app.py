import io
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import mne
from scipy.signal import welch, spectrogram  # <-- added spectrogram
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
import joblib
import plotly.graph_objects as go
import plotly.express as px
import tempfile  # <-- existing
import matplotlib.pyplot as plt  # <-- for optional topomaps

st.set_page_config(page_title="EEG Emotion Detection (EDF)", layout="wide")

EMOTIONS = ["happy", "sad", "neutral", "angry"]

# -------------------------------
# Utility: Feature Extraction
# -------------------------------

def bandpower_welch(data, sf, fmin, fmax):
    """Compute bandpower via Welch per epoch (channels x samples). Returns (channels,)."""
    nperseg = min(256, data.shape[-1])
    freqs, psd = welch(data, sf, nperseg=nperseg, axis=-1)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    bp = np.trapz(psd[..., idx], freqs[idx], axis=-1)
    return bp  # (channels,)

def compute_epoch_features(epoch_data, sf, bands=None):
    """
    epoch_data: np.ndarray shape (n_channels, n_samples)
    Returns 1D feature vector.
    """
    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }
    feats = []

    # Band powers per channel
    band_powers = {}
    for bname, (f1, f2) in bands.items():
        band_powers[bname] = bandpower_welch(epoch_data, sf, f1, f2)  # (channels,)

    # Stack per-channel bandpowers
    for bname in bands.keys():
        feats.append(band_powers[bname])

    # Ratios (alpha/beta, beta/(alpha+theta))
    alpha = band_powers["alpha"] + 1e-12
    beta = band_powers["beta"] + 1e-12
    theta = band_powers["theta"] + 1e-12
    feats.append(alpha / beta)
    feats.append(beta / (alpha + theta))

    # Hjorth parameters per channel
    x = epoch_data
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var_x = np.var(x, axis=-1) + 1e-12
    var_dx = np.var(dx, axis=-1) + 1e-12
    var_ddx = np.var(ddx, axis=-1) + 1e-12
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / (mobility + 1e-12)
    feats.append(var_x)
    feats.append(mobility)
    feats.append(complexity)

    # Differential entropy per band per channel (using band power as variance proxy)
    for bname in bands.keys():
        var_proxy = band_powers[bname] + 1e-12
        de = 0.5 * np.log(2 * np.pi * np.e * var_proxy)
        feats.append(de)

    # Aggregate channel statistics to reduce dimensionality (mean, std, median)
    feats = np.stack(feats, axis=0)  # (feature_groups, channels)
    stats = [np.mean(feats, axis=1), np.std(feats, axis=1), np.median(feats, axis=1)]
    stats = np.concatenate(stats, axis=0)  # (feature_groups*3, )
    return stats.astype(np.float32)

def extract_features_from_raw(raw, picks, sf_target, epoch_len_s, epoch_overlap_s):
    """
    Returns:
        X: (n_epochs, n_features)
        t_starts: list of start times for epochs
    """
    if picks is None or len(picks) == 0:
        picks = mne.pick_types(raw.info, eeg=True)

    # Resample
    if sf_target is not None:
        raw_res = raw.copy().resample(sf_target)
        sf = sf_target
    else:
        raw_res = raw.copy()
        sf = raw.info["sfreq"]

    data = raw_res.get_data(picks=picks)  # (n_channels, n_samples)
    n_channels, n_samples = data.shape
    step = int((epoch_len_s - epoch_overlap_s) * sf)
    win = int(epoch_len_s * sf)
    if win <= 0 or step <= 0 or win > n_samples:
        raise ValueError("Bad epoch/window parameters.")

    X = []
    t_starts = []
    for start in range(0, n_samples - win + 1, step):
        epoch = data[:, start:start + win]
        feats = compute_epoch_features(epoch, sf)
        X.append(feats)
        t_starts.append(start / sf)

    X = np.vstack(X) if len(X) else np.empty((0, 1))
    return X, t_starts

def default_heuristic_classifier(X):
    """
    Fallback classifier if user hasn't provided a trained model.
    Returns proba-like scores and labels (demo only).
    """
    if X.size == 0:
        return np.array([]), np.array([])
    scaler = (X - X.mean(0)) / (X.std(0) + 1e-9)
    score = scaler @ np.ones((scaler.shape[1],)) / scaler.shape[1]  # crude 1D score

    labels = []
    for s in score:
        if s > 0.7:
            labels.append("happy")
        elif s > 0.2:
            labels.append("neutral")
        elif s > -0.5:
            labels.append("sad")
        else:
            labels.append("angry")
    labels = np.array(labels)

    proba = np.zeros((len(labels), len(EMOTIONS)), dtype=float)
    for i, lab in enumerate(labels):
        j = EMOTIONS.index(lab)
        proba[i, j] = 0.7
        proba[i, :] += 0.3 / len(EMOTIONS)
    return proba, labels

def majority_vote(labels):
    if len(labels) == 0:
        return None
    vals, counts = np.unique(labels, return_counts=True)
    return vals[np.argmax(counts)]

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")
epoch_len_s = st.sidebar.number_input("Epoch length (s)", 1.0, 10.0, 2.0, 0.5)
epoch_overlap_s = st.sidebar.number_input("Epoch overlap (s)", 0.0, 9.5, 0.5, 0.5)
sf_target = st.sidebar.number_input("Resample to (Hz)", 64, 512, 128, 32)
l_freq = st.sidebar.number_input("Bandpass low (Hz)", 0.1, 20.0, 1.0, 0.1)
h_freq = st.sidebar.number_input("Bandpass high (Hz)", 30.0, 100.0, 45.0, 1.0)
notch = st.sidebar.selectbox("Notch filter", ["None", "50Hz", "60Hz"], index=1)
ref_type = st.sidebar.selectbox("Re-reference", ["None", "Average"], index=1)

# Extra controls for visualizations
st.sidebar.subheader("🖼️ Visuals")
preview_len_s = st.sidebar.slider("Preview window (s)", 3, 20, 10, 1)
spec_nperseg = st.sidebar.select_slider("Spectrogram window (samples)", options=[64,128,256,512,1024], value=256)
show_topomaps = st.sidebar.checkbox("Show topographic bandpower maps (if montage available)", value=True)

st.title("EEG Emotion Detection")

colL, colR = st.columns([2, 1])
with colL:
    edf_file = st.file_uploader("Upload EDF file", type=["edf"], accept_multiple_files=False)
with colR:
    model_file = st.file_uploader("Optional: upload pretrained classifier (.pkl)", type=["pkl"])

st.markdown(
    """
- Upload an **EDF** file (multi-channel EEG).
- Optionally upload a **pretrained scikit-learn classifier** saved with `joblib.dump(model, "model.pkl")`.
- The app will preprocess, extract features per epoch, and predict one of **happy / sad / neutral / angry**.
"""
)

# -------------------------------
# Main logic
# -------------------------------
if edf_file is not None:
    # --- Load EDF via a temporary file path (required by MNE on Windows) ---
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(edf_file.getvalue())
            tmp_path = tmp.name

        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)

        # Basic cleaning
        if notch != "None":
            freq = 50.0 if notch == "50Hz" else 60.0
            raw.notch_filter(freqs=[freq], verbose=False)
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

        # Re-reference
        if ref_type == "Average":
            raw.set_eeg_reference("average", verbose=False)

        # Pick EEG channels only
        picks = mne.pick_types(raw.info, eeg=True)

        st.subheader("📊 Signal Summary")
        info = {
            "n_channels": len(picks),
            "sfreq": float(raw.info["sfreq"]),
            "duration_s": float(raw.n_times / raw.info["sfreq"]),
            "channels": [raw.ch_names[i] for i in picks][:20]  # show first 20
        }
        st.json(info)

        # ------------ NEW: Interactive Signal Visualizations ------------
        st.subheader("🧭 Channel Explorer & Signal Views")

        # Channel selector + time window selector
        ch_names = [raw.ch_names[i] for i in picks]
        sel_col1, sel_col2 = st.columns([1,1])
        with sel_col1:
            sel_ch = st.selectbox("Select channel", ch_names, index=0)
        with sel_col2:
            start_time = st.number_input(
                "Start time (s)", min_value=0.0,
                max_value=max(0.0, float(info["duration_s"] - preview_len_s)),
                value=0.0, step=1.0
            )

        sf = raw.info["sfreq"]
        ch_idx = picks[ch_names.index(sel_ch)]
        i0 = int(start_time * sf)
        i1 = int(min(raw.n_times, i0 + preview_len_s * sf))
        seg = raw.get_data(picks=[ch_idx])[:, i0:i1][0]
        t = np.arange(seg.shape[0]) / sf + start_time

        # Raw trace for selected channel
        trace_fig = go.Figure()
        trace_fig.add_trace(go.Scatter(x=t, y=seg, mode='lines', name=sel_ch))
        trace_fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10),
                                xaxis_title="Time (s)", yaxis_title="Amplitude (µV)")
        st.plotly_chart(trace_fig, use_container_width=True)

        # Spectrogram (time-frequency) for selected channel
        st.caption("Spectrogram (selected channel)")
        f_spec, t_spec, Sxx = spectrogram(seg, fs=sf, nperseg=int(spec_nperseg), noverlap=int(spec_nperseg//2))
        # Limit to, say, 1–45 Hz for readability
        keep = (f_spec >= 1) & (f_spec <= 45)
        f_spec = f_spec[keep]
        Sxx = Sxx[keep, :]
        spec_fig = px.imshow(10*np.log10(Sxx + 1e-12), origin="lower",
                             labels=dict(x="Time bins", y="Frequency (Hz)", color="Power (dB)"),
                             aspect="auto")
        # Add proper axis ticks
        spec_fig.update_yaxes(tickmode="array", tickvals=np.linspace(0, len(f_spec)-1, 6),
                              ticktext=[f"{v:.0f}" for v in np.linspace(f_spec[0], f_spec[-1], 6)])
        st.plotly_chart(spec_fig, use_container_width=True)

        # PSD (Welch) for multiple channels (up to 8 to avoid clutter)
        st.caption("Power Spectral Density (Welch)")
        max_show = min(8, len(picks))
        nperseg = min(1024, int(sf*2))
        psd_fig = go.Figure()
        freqs, psd_all = None, []
        for i in range(max_show):
            ch = picks[i]
            data = raw.get_data(picks=[ch])[0]
            freqs, psd = welch(data, fs=sf, nperseg=nperseg)
            psd_all.append(psd)
            psd_fig.add_trace(go.Scatter(x=freqs, y=10*np.log10(psd + 1e-12),
                                         mode="lines", name=raw.ch_names[ch], opacity=0.7))
        if len(psd_all):
            psd_all = np.vstack(psd_all)
            psd_fig.add_trace(go.Scatter(
                x=freqs, y=10*np.log10(psd_all.mean(axis=0)+1e-12),
                mode="lines", name="Mean(PSD)", line=dict(width=3)
            ))
        psd_fig.update_layout(height=250, xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)")
        st.plotly_chart(psd_fig, use_container_width=True)

        # Bandpower bars per channel for the preview window
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }
        # compute on preview segment for all shown channels
        bp_rows = []
        for i in range(max_show):
            ch = picks[i]
            data = raw.get_data(picks=[ch])[0][i0:i1]
            for bname, (f1,f2) in bands.items():
                # pad to avoid too-short segments
                if len(data) < 128:
                    continue
                freqs_bp, psd_bp = welch(data, fs=sf, nperseg=min(256, len(data)))
                idx = (freqs_bp >= f1) & (freqs_bp <= f2)
                val = np.trapz(psd_bp[idx], freqs_bp[idx]) if idx.any() else 0.0
                bp_rows.append({"channel": raw.ch_names[ch], "band": bname, "power": float(val)})
        if bp_rows:
            bp_df = pd.DataFrame(bp_rows)
            bp_fig = px.bar(bp_df, x="band", y="power", color="channel", barmode="group",
                            title="Bandpower (preview window)")
            bp_fig.update_layout(height=280, xaxis_title="", yaxis_title="Power (a.u.)")
            st.plotly_chart(bp_fig, use_container_width=True)

        # Optional: Topographic maps of bandpower over the whole recording (or preview) if montage present
        if show_topomaps:
            can_topomap = False
            try:
                # MNE needs channel positions; many EDFs include standard montages
                # If montage is missing, this may fail; we catch and skip gracefully.
                pos = mne.channels.layout._find_topomap_coords(raw.info, picks=picks)  # private util; just to check
                can_topomap = True
            except Exception:
                can_topomap = False

            if can_topomap:
                st.caption("Topographic Bandpower (whole recording)")
                # compute bandpower per channel over full (filtered) recording
                dat_full = raw.get_data(picks=picks)
                topo_cols = st.columns(3)
                topo_bands = [("theta",(4,8)), ("alpha",(8,13)), ("beta",(13,30))]
                for (bname, (f1,f2)), cc in zip(topo_bands, topo_cols):
                    vals = bandpower_welch(dat_full, sf, f1, f2)  # (channels,)
                    fig_, ax_ = plt.subplots(figsize=(3.8, 3.2))
                    mne.viz.plot_topomap(vals, raw.info, axes=ax_, show=False, names=None, contours=6, sensors=False)
                    ax_.set_title(f"{bname.capitalize()} power")
                    with cc:
                        st.pyplot(fig_, clear_figure=True)
            else:
                st.info("Topomap skipped: channel positions/montage not found in this EDF.")

        # ---------------- Existing quick preview (kept) ----------------
        with st.expander("Preview raw (first ~10s)"):
            sf = raw.info["sfreq"]
            seg = raw.get_data(picks=picks)[:, :int(min(10*sf, raw.n_times))]
            t = np.arange(seg.shape[1]) / sf
            fig = go.Figure()
            offset = 0
            for i in range(min(seg.shape[0], 8)):  # show up to 8 channels
                fig.add_trace(go.Scatter(x=t, y=seg[i] + offset, name=raw.ch_names[picks[i]], mode='lines'))
                offset += 100.0  # visual offset
            fig.update_layout(height=300, showlegend=True, xaxis_title="Time (s)", yaxis_title="Amplitude (µV, offset)")
            st.plotly_chart(fig, use_container_width=True)

        # ---------------- Feature extraction & Prediction ---------------
        with st.spinner("Extracting features..."):
            X, t_starts = extract_features_from_raw(
                raw, picks=picks, sf_target=sf_target,
                epoch_len_s=epoch_len_s, epoch_overlap_s=epoch_overlap_s
            )

        st.write(f"Extracted **{X.shape[0]} epochs** with **{X.shape[1]} features** each.")

        # Load model if provided
        model = None
        scaler = None
        if model_file is not None:
            try:
                bundle = joblib.load(model_file)
                if isinstance(bundle, dict) and "model" in bundle:
                    model = bundle["model"]
                    scaler = bundle.get("scaler", None)
                else:
                    model = bundle
            except Exception as e:
                st.error(f"Failed to load model: {e}")

        # Scale if needed
        X_infer = X.copy()
        if scaler is not None:
            try:
                check_is_fitted(scaler)
                X_infer = scaler.transform(X_infer)
            except Exception:
                scaler.fit(X_infer)
                X_infer = scaler.transform(X_infer)

        # Predict
        if model is not None:
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_infer)
                    labels_idx = np.argmax(proba, axis=1)
                else:
                    labels_idx = model.predict(X_infer)
                    proba = np.zeros((len(labels_idx), len(EMOTIONS)))
                    proba[np.arange(len(labels_idx)), labels_idx] = 1.0

                # Ensure classifier classes_ are aligned or remap
                if hasattr(model, "classes_"):
                    classes = list(model.classes_)
                    if set(classes) == set(EMOTIONS):
                        reorder = [classes.index(e) for e in EMOTIONS]
                        proba = proba[:, reorder]
                        labels_idx = np.argmax(proba, axis=1)

                labels = np.array([EMOTIONS[i] for i in labels_idx])
            except Exception as e:
                st.warning(f"Model prediction failed ({e}). Using heuristic fallback.")
                proba, labels = default_heuristic_classifier(X_infer)
        else:
            st.info("No model provided. Using a simple heuristic fallback (for demo only).")
            proba, labels = default_heuristic_classifier(X_infer)

        if len(labels) == 0:
            st.error("No predictions to show (no epochs?). Check window settings.")
            st.stop()

        # Majority vote
        overall = majority_vote(labels)
        st.subheader(f"🧠 Overall predicted emotion: **{overall}**")

        # Timeline
        with st.expander("Prediction timeline"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_starts, y=[EMOTIONS.index(l) for l in labels],
                                     mode="lines+markers", name="Prediction"))
            fig.update_yaxes(tickmode="array", tickvals=list(range(len(EMOTIONS))), ticktext=EMOTIONS)
            fig.update_layout(height=300, xaxis_title="Time (s)", yaxis_title="Emotion")
            st.plotly_chart(fig, use_container_width=True)

        # Probability bar chart (mean over time)
        with st.expander("Average class probabilities"):
            mean_proba = proba.mean(axis=0)
            fig = go.Figure(go.Bar(x=EMOTIONS, y=mean_proba))
            fig.update_layout(height=300, yaxis_title="Mean probability")
            st.plotly_chart(fig, use_container_width=True)

        # Download CSV of per-epoch predictions
        out_df = pd.DataFrame({
            "t_start_s": t_starts,
            "prediction": labels
        })
        for j, e in enumerate(EMOTIONS):
            out_df[f"proba_{e}"] = proba[:, j]
        st.download_button(
            "Download per-epoch predictions (CSV)",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="eeg_emotion_predictions.csv",
            mime="text/csv"
        )
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
else:
    st.info("Upload an EDF file to begin.")
