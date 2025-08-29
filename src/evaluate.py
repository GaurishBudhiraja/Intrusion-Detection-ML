# evaluate.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

SEQ_LEN = 30

# 1) AE detection metrics
def eval_ae():
    print("Evaluating GRU-AE detection...")
    ae = load_model(os.path.join(MODEL_DIR, "gru_ae.h5"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "gru_ae_scaler.pkl"))
    thr = None
    import json
    thr = json.load(open(os.path.join(MODEL_DIR, "gru_ae_threshold.json")))["threshold"]

    # prepare test mixture: normal (some) + faults (some)
    normal = pd.read_csv(os.path.join(DATA_DIR, "gru_ae_normal.csv")).sample(n=5000, random_state=1)
    faults = pd.read_csv(os.path.join(DATA_DIR, "faults.csv")).sample(n=5000, random_state=1)
    Xn = normal[[f"p{i}" for i in range(SEQ_LEN)]].values
    snrn = normal["snr"].values.reshape(-1,1)
    Xn_full = np.stack([Xn, np.repeat(snrn, SEQ_LEN, axis=1)], axis=-1)
    nf = Xn_full.reshape(Xn_full.shape[0]*SEQ_LEN,2)
    Xn_s = scaler.transform(nf).reshape(Xn_full.shape)
    Xf = faults[[f"p{i}" for i in range(SEQ_LEN)]].values
    snrf = faults["snr"].values.reshape(-1,1)
    Xf_full = np.stack([Xf, np.repeat(snrf, SEQ_LEN, axis=1)], axis=-1)
    Xf_s = scaler.transform(Xf_full.reshape(Xf_full.shape[0]*SEQ_LEN,2)).reshape(Xf_full.shape)

    # recon errors
    recon_n = ae.predict(Xn_s, batch_size=256)
    err_n = np.mean((recon_n - Xn_s)**2, axis=(1,2))
    recon_f = ae.predict(Xf_s[:5000], batch_size=256)
    err_f = np.mean((recon_f - Xf_s[:5000])**2, axis=(1,2))
    y_true = np.concatenate([np.zeros_like(err_n), np.ones_like(err_f)])
    y_pred = (np.concatenate([err_n, err_f]) > thr).astype(int)
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, np.concatenate([err_n, err_f]))
    print("Detection — Precision, Recall, F1:", p, r, f, "AUC:", auc)

# 2) BiGRU evaluation
def eval_bigrul():
    print("Evaluating Attention-BiGRU (diagnose+localize)...")
    model = load_model(os.path.join(MODEL_DIR, "abigrul_final.h5"), compile=False)
    scaler = joblib.load(os.path.join(MODEL_DIR, "abigrul_scaler.pkl"))
    faults = pd.read_csv(os.path.join(DATA_DIR, "faults.csv"))
    X = faults[[f"p{i}" for i in range(SEQ_LEN)]].values
    snr = faults["snr"].values.reshape(-1,1)
    X_full = np.stack([X, np.repeat(snr, SEQ_LEN, axis=1)], axis=-1)
    n,t,f = X_full.shape
    Xs = scaler.transform(X_full.reshape(n*t,f)).reshape(n,t,f)
    y_cls = faults["fault_type"].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y_cls)
    y_num = le.transform(y_cls)
    # split small test portion
    from sklearn.model_selection import train_test_split
    X_tr, X_te, ytr, yte = train_test_split(Xs, y_num, test_size=0.2, random_state=42, stratify=y_num)
    pos = faults["position"].values
    _, pos_te = train_test_split(pos, test_size=0.2, random_state=42, stratify=y_num)

    preds = model.predict(X_te, batch_size=256)
    y_pred_cls = np.argmax(preds[0], axis=1)
    y_pred_pos = preds[1].ravel()

    print("Classification report:")
    print(classification_report(yte, y_pred_cls, target_names=le.classes_, zero_division=0))
    # per-attack accuracy (for bar chart)
    from collections import defaultdict
    acc = defaultdict(list)
    for gt, pr in zip(yte, y_pred_cls):
        acc[le.classes_[gt]].append(int(gt==pr))
    attack_acc = {k: np.mean(v) for k,v in acc.items()}
    print("Per-attack accuracy:", attack_acc)
    # plot barevaluate.py
    plt.figure(figsize=(8,4))
    plt.bar(list(attack_acc.keys()), [v*100 for v in attack_acc.values()])
    plt.ylabel("Accuracy (%)")
    plt.title("Detection Accuracy by Attack Type")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # localization RMSE
    rmse = np.sqrt(mean_squared_error(pos_te, y_pred_pos))
    print("Localization RMSE (m):", rmse)

if __name__ == "__main__":
    eval_ae()
    eval_bigrul()
