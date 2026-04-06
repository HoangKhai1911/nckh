import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os
import glob
from datetime import datetime

import mne
from scipy.signal import butter, lfilter, welch
from scipy.stats import entropy

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

# THÊM MỚI: Thư viện cho SMOTE
from imblearn.over_sampling import SMOTE

# ==================================
# ⚙️ CONFIGURATION
# ==================================

class CONFIG:
    SEED = 42
    RAW_DATA_DIR = r"A:\lstm+cnn\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette" #sửa dòng này
    PROCESSED_DATA_DIR = "./processed_data_optimized" # Đổi tên cho dữ liệu mới
    
    MODEL_DIR = "./saved_models_rf_optimized" 
    PLOTS_DIR = "./visualization_plots_rf_optimized"
    SLEEP_STAGE_LABELS = ["Wake", "N1", "N2", "N3", "REM"]
    ANNOTATION_MAP = {
        "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
        "Sleep stage 3": 3, "Sleep stage 4": 3,
        "Sleep stage R": 4, "Sleep stage ?": -1, "Movement time": -1
    }
    EPOCH_DURATION_S = 30
    FREQ_BANDS = {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 12], "sigma": [12, 16], "beta": [16, 30]}

# ==================================
# 🔬 DATA PREPROCESSING (TỐI ƯU HÓA)
# ==================================
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def get_spectral_features(epoch_data, fs):
    freqs, psd = welch(epoch_data, fs=fs, nperseg=fs*2)
    band_powers = []
    for band in CONFIG.FREQ_BANDS.values():
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        band_powers.append(np.sum(psd[idx_band]))
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    spectral_entropy = entropy(psd_norm)
    return band_powers + [spectral_entropy]

def hjorth_parameters(epoch_data):
    """Tính toán Hjorth Parameters: Activity, Mobility, Complexity."""
    activity = np.var(epoch_data)
    diff1 = np.diff(epoch_data)
    diff2 = np.diff(diff1)
    mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
    complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility if np.var(diff1) > 0 and mobility > 0 else 0
    return [mobility, complexity]

def extract_features(epoch_data, fs):
    """Trích xuất đặc trưng bao gồm cả Hjorth parameters."""
    stat_features = [np.std(epoch_data), np.ptp(epoch_data)]
    spectral_features = get_spectral_features(epoch_data, fs)
    hjorth_features = hjorth_parameters(epoch_data) # THÊM MỚI
    return stat_features + spectral_features + hjorth_features

def preprocess_raw_edf_data(raw_data_path):
    print("🔬 Starting raw EDF data preprocessing with Hjorth & Entropy features...")
    psg_files = sorted(glob.glob(os.path.join(raw_data_path, "*PSG.edf")))
    hypno_files = sorted(glob.glob(os.path.join(raw_data_path, "*Hypnogram.edf")))
    if not psg_files or not hypno_files or len(psg_files) != len(hypno_files): return None, None, None
    all_features, all_labels, all_subject_ids = [], [], []
    for psg_filepath, hypno_filepath in zip(psg_files, hypno_files):
        subject_id = os.path.basename(psg_filepath).split('-')[0]
        print(f"   -> Processing subject: {subject_id}")
        raw = mne.io.read_raw_edf(psg_filepath, preload=True, verbose='WARNING')
        annot = mne.read_annotations(hypno_filepath)
        raw.set_annotations(annot, emit_warning=False)
        eeg_channel, fs = 'EEG Fpz-Cz', int(raw.info['sfreq'])
        eeg_data = raw.get_data(picks=[eeg_channel])[0]
        eeg_filtered = bandpass_filter(eeg_data, lowcut=0.5, highcut=45.0, fs=fs)
        events, _ = mne.events_from_annotations(raw, event_id=CONFIG.ANNOTATION_MAP, chunk_duration=CONFIG.EPOCH_DURATION_S)
        for event in events:
            start_sample, _, label = event
            if label == -1: continue
            end_sample = start_sample + CONFIG.EPOCH_DURATION_S * fs
            if end_sample > len(eeg_filtered): continue
            epoch_segment = eeg_filtered[start_sample:end_sample]
            features = extract_features(epoch_segment, fs)
            all_features.append(features)
            all_labels.append(label)
            all_subject_ids.append(subject_id)
    print("✅ Raw data preprocessing complete.")
    return np.array(all_features), np.array(all_labels), np.array(all_subject_ids)

def load_data():
    os.makedirs(CONFIG.PROCESSED_DATA_DIR, exist_ok=True)
    X_path = os.path.join(CONFIG.PROCESSED_DATA_DIR, "X_features.npy")
    y_path = os.path.join(CONFIG.PROCESSED_DATA_DIR, "y_labels.npy")
    subjects_path = os.path.join(CONFIG.PROCESSED_DATA_DIR, "subject_ids.npy")
    try:
        print("📥 Attempting to load pre-processed data...")
        X, y, subject_ids = np.load(X_path), np.load(y_path), np.load(subjects_path)
        print(f"✅ Pre-processed data loaded successfully: X{X.shape}, y{y.shape}")
        return X, y, subject_ids
    except FileNotFoundError:
        print("❌ Pre-processed data not found. Processing from raw EDF files...")
        X, y, subject_ids = preprocess_raw_edf_data(CONFIG.RAW_DATA_DIR)
        if X is not None and len(X) > 0:
            print("💾 Saving processed data for future use...")
            np.save(X_path, X)
            np.save(y_path, y)
            np.save(subjects_path, subject_ids)
            print("✅ Processed data saved.")
        return X, y, subject_ids

# ==================================
# 🤖 MODEL TRAINING (TÍCH HỢP SMOTE)
# ==================================

def train_random_forest_model(X, y, subject_ids):
    print("🤖 Training Optimized Random Forest with SMOTE...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    unique_subjects = np.unique(subject_ids)
    n_splits = min(7, len(unique_subjects))
    if n_splits < 2: return [None]*8
    gkf = GroupKFold(n_splits=n_splits)
    
    # SMOTE sẽ xử lý việc mất cân bằng, không cần class_weight nữa
    smote = SMOTE(random_state=CONFIG.SEED)
    
    accuracies, f1_scores, kappa_scores, all_y_true, all_y_pred = [], [], [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=subject_ids)):
        print(f"🔁 Fold {fold + 1}")
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # ÁP DỤNG SMOTE CHỈ TRÊN DỮ LIỆU HUẤN LUYỆN
        print("   -> Applying SMOTE to balance training data...")
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"   -> Original training size: {len(X_train)}, Resampled size: {len(X_train_res)}")

        # Huấn luyện trên dữ liệu đã được cân bằng
        model = RandomForestClassifier(n_estimators=200, random_state=CONFIG.SEED, n_jobs=-1)
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        accuracies.append(accuracy); f1_scores.append(f1); kappa_scores.append(kappa)
        all_y_true.extend(y_test.tolist()); all_y_pred.extend(y_pred.tolist())
        print(f"   ✅ Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
    
    # Huấn luyện mô hình cuối cùng trên toàn bộ dữ liệu đã được cân bằng
    print("\n🌳 Training final model on the entire balanced dataset...")
    X_scaled_res, y_res = smote.fit_resample(X_scaled, y)
    final_model = RandomForestClassifier(n_estimators=200, random_state=CONFIG.SEED, n_jobs=-1)
    final_model.fit(X_scaled_res, y_res)
    
    overall_accuracy, overall_f1, overall_kappa = np.mean(accuracies), np.mean(f1_scores), np.mean(kappa_scores)
    
    print(f"\n🎯 Final Model Performance:")
    print(f"   Accuracy: {overall_accuracy:.4f}")
    print(f"   F1-Score: {overall_f1:.4f}")
    print(f"   Cohen's Kappa: {overall_kappa:.4f}")
    
    return final_model, scaler, all_y_true, all_y_pred, overall_accuracy, overall_f1, overall_kappa

# ==================================
# 🚀 MAIN PIPELINE
# ==================================
def main():
    print("🚀 SLEEP STAGE CLASSIFICATION PIPELINE (OPTIMIZED RF with SMOTE)")
    print("=" * 50)
    
    X, y, subject_ids = load_data()
    if X is None: return

    results = train_random_forest_model(X, y, subject_ids)
    if results[0] is None: return 
    
    model, scaler, all_y_true, all_y_pred, acc, f1, kappa = results
    
    print("\n📋 DETAILED CLASSIFICATION REPORT:")
    print("=" * 40)
    final_kappa = cohen_kappa_score(all_y_true, all_y_pred)
    print(classification_report(all_y_true, all_y_pred, target_names=CONFIG.SLEEP_STAGE_LABELS, digits=4, zero_division=0))
    print(f"Overall Cohen's Kappa: {final_kappa:.4f}")
    
    print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()