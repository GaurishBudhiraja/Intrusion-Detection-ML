# infer.py
import os, joblib, numpy as np, pandas as pd
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
SEQ_LEN = 30

def load_models():
    ae = load_model(os.path.join(MODEL_DIR, "gru_ae.h5"))
    ae_scaler = joblib.load(os.path.join(MODEL_DIR, "gru_ae_scaler.pkl"))
    thr = __import__("json").load(open(os.path.join(MODEL_DIR, "gru_ae_threshold.json")))["threshold"]
    abigrul = load_model(os.path.join(MODEL_DIR, "abigrul_final.h5"), compile=False)
    abigrul_scaler = joblib.load(os.path.join(MODEL_DIR, "abigrul_scaler.pkl"))
    return ae, ae_scaler, thr, abigrul, abigrul_scaler

def preprocess_seq(seq, snr, scaler):
    # seq: length-30 array, snr: scalar
    x = np.array(seq).reshape(1, SEQ_LEN)
    snr_rep = np.repeat(np.array([[snr]]), SEQ_LEN, axis=1)
    X_full = np.stack([x, snr_rep], axis=-1)
    n,t,f = X_full.shape
    Xs = scaler.transform(X_full.reshape(n*t,f)).reshape(n,t,f)
    return Xs

def infer_one(seq, snr):
    ae, ae_scaler, thr, abigrul, abigrul_scaler = load_models()
    Xs = preprocess_seq(seq, snr, ae_scaler)
    recon = ae.predict(Xs)
    err = np.mean((recon - Xs)**2)
    if err <= thr:
        return {"anomaly": False, "score": float(err)}
    # anomaly -> diagnose and localize
    Xs2 = preprocess_seq(seq, snr, abigrul_scaler)
    pred_cls, pred_pos = abigrul.predict(Xs2)
    cls_idx = int(np.argmax(pred_cls, axis=1)[0])
    from sklearn.preprocessing import LabelEncoder
    # we don't persist label encoder; assume order or load mapping externally
    return {"anomaly": True, "score": float(err), "class_idx": cls_idx, "position_est": float(pred_pos[0,0])}

if __name__ == "__main__":
    # demo using a sample from datasets
    faults = pd.read_csv(os.path.join(DATA_DIR, "faults.csv"))
    sample = faults.iloc[0]
    seq = sample[[f"p{i}" for i in range(SEQ_LEN)]].values
    snr = float(sample["snr"])
    res = infer_one(seq, snr)
    print(res)
