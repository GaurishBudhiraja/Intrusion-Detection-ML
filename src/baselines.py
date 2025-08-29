# src/baselines.py
import os, numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "datasets")
SEQ_LEN = 30

def physical_threshold_ids(df, thr_osnr=8.0, thr_trace_err=0.002):
    """Predict attack if snr<thr OR trace roughness>thr."""
    seq = df[[f"p{i}" for i in range(SEQ_LEN)]].values
    snr = df["snr"].values
    rough = (np.abs(np.diff(seq, axis=1))).mean(axis=1)
    pred = ((snr < thr_osnr) | (rough > thr_trace_err)).astype(int)
    y = (df["label"]!="normal").astype(int).values
    p,r,f,_ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    auc = roc_auc_score(y, (-(snr-thr_osnr)) + (rough-thr_trace_err))
    return {"Accuracy":float((pred==y).mean()*100),
            "Precision":float(p*100),"Recall":float(r*100),"F1":float(f*100),
            "FPR":float((pred>y).mean()*100),"AUC":float(auc)}

def rules_ids(df):
    """Crude heuristic to mimic signature IDS on network features."""
    X = df[[f"net_{i}" for i in range(10)]].values
    y = (df["label"]!="normal").astype(int).values

    mean_pkt, std_pkt, mean_iat, std_iat, flow_dur, syn_ratio, fin_ratio, bpf, pe, ie = X.T

    score = np.zeros(len(df))
    score += (syn_ratio > 0.11) * 1.0
    score += (std_pkt   > 180)  * 0.6
    score += (mean_iat  > 18)   * 0.8
    score += (pe        > 3.0)  * 0.7
    score += (ie        > 3.0)  * 0.7
    pred = (score >= 1.0).astype(int)

    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    p,r,f,_ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    auc = roc_auc_score(y, score)
    return {"Accuracy":float((pred==y).mean()*100),
            "Precision":float(p*100),"Recall":float(r*100),"F1":float(f*100),
            "FPR":float((pred>y).mean()*100),"AUC":float(auc)}
