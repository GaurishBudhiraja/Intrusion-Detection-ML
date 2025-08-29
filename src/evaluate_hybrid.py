# src/evaluate_hybrid.py
import os, json, numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt, seaborn as sns, psutil, time
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf
from baselines import physical_threshold_ids, rules_ids

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
SEQ_LEN=30

# ---- FIXED AttentionLayer ----
# ---- FIXED AttentionLayer (match fusion_train.py) ----
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
            name="W_att"
        )
        self.u = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
            name="u_att"
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time, features)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1))   # (batch,time,features)
        ait = tf.nn.softmax(tf.tensordot(uit, self.u, axes=1), axis=1)  # (batch,time,1)
        return tf.reduce_sum(inputs * ait, axis=1)            # (batch,features)
# -------------------------------------------------------

def _load_eval_data(csv="hybrid.csv"):
    df = pd.read_csv(os.path.join(DATA_DIR, csv))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    return df.iloc[int(0.8*n):].reset_index(drop=True)

def _prep_fusion_batch(df):
    Xp = df[[f"p{i}" for i in range(SEQ_LEN)]].values.astype(np.float32)
    snr = df["snr"].values.astype(np.float32).reshape(-1,1)
    X_opt = np.stack([Xp, np.repeat(snr, SEQ_LEN, axis=1)], axis=-1)

    X_net = df[[f"net_{i}" for i in range(10)]].values.astype(np.float32)

    opt_scaler = joblib.load(os.path.join(MODEL_DIR, "fusion_opt_scaler.pkl"))
    net_scaler = joblib.load(os.path.join(MODEL_DIR, "fusion_net_scaler.pkl"))

    n,t,c = X_opt.shape
    Xo = opt_scaler.transform(X_opt.reshape(-1,c)).reshape(n,t,c)
    Xn = net_scaler.transform(X_net)
    return Xo, Xn

def eval_fusion():
    df = _load_eval_data()
    Xo, Xn = _prep_fusion_batch(df)
    model = load_model(
        os.path.join(MODEL_DIR, "fusion_final.h5"),
        compile=False,
        custom_objects={"AttentionLayer": AttentionLayer}
    )
    label_map = json.load(open(os.path.join(MODEL_DIR, "fusion_labels.json")))["classes"]
    y_true = df["label"].values
    y_num = np.array([label_map.index(l) for l in y_true])

    t0=time.time(); preds = model.predict([Xo,Xn], batch_size=256, verbose=0); t1=time.time()
    cls = np.argmax(preds[0], axis=1); pos = preds[1].ravel()
    prob_attack = 1.0 - preds[0][:, label_map.index("normal")]

    report = classification_report(y_num, cls, target_names=label_map, zero_division=0)
    auc = roc_auc_score((y_true!="normal").astype(int), prob_attack)
    acc = {lab: float(np.mean(cls[y_num==i]==i)*100) for i,lab in enumerate(label_map) if lab!="normal"}

    print("\n=== Fusion Model — Classification Report ===")
    print(report)
    print("AUC (attack vs normal):", round(float(auc), 4))
    print("Per-attack accuracy (%):", {k: round(v,2) for k,v in acc.items()})

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve((y_true!="normal").astype(int), prob_attack)
    plt.figure(figsize=(5,4))
    plt.plot(fpr,tpr,label=f"Proposed (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'--',label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR,"roc_fusion.png"))

    labs = [k for k in acc.keys()]
    vals = [acc[k] for k in labs]
    plt.figure(figsize=(7,4))
    plt.bar(labs, vals)
    plt.ylabel("Accuracy (%)")
    plt.title("Detection Accuracy by Attack Type")
    plt.xticks(rotation=25)
    plt.tight_layout(); plt.savefig(os.path.join(MODEL_DIR,"attack_bar.png"))

    proc = psutil.Process()
    mem_mb = proc.memory_info().rss/1024/1024
    avg_ms = (t1-t0)/len(df)*1000
    print(f"\nLatency per sample (avg): {avg_ms:.2f} ms | Process RSS ~ {mem_mb:.0f} MB")

def eval_baselines():
    df = _load_eval_data()
    print("\n=== Physical Threshold IDS ===")
    print(physical_threshold_ids(df))
    print("\n=== Rules IDS (Snort/Suricata stand-in) ===")
    print(rules_ids(df))

if __name__ == "__main__":
    eval_fusion()
    eval_baselines()
