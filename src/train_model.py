# train.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Bidirectional, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN = 30

# -------------------------
# Attention Layer (simple)
# -------------------------
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True, name="W_att")
        self.u = self.add_weight(shape=(input_shape[-1],1),
                                 initializer="glorot_uniform", trainable=True, name="u_att")
        super(AttentionLayer, self).build(input_shape)
    def call(self, inputs):
        # inputs: (batch, time, features)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1))  # (batch, time, features)
        ait = tf.nn.softmax(tf.tensordot(uit, self.u, axes=1), axis=1)  # (batch, time, 1)
        weighted = tf.reduce_sum(inputs * ait, axis=1)  # (batch, features)
        return weighted

# -------------------------
# 1) Train GRU Autoencoder (GRU-AE)
# -------------------------
def train_gru_ae():
    print("Loading normal data for GRU-AE...")
    normal = pd.read_csv(os.path.join(DATA_DIR, "gru_ae_normal.csv"))
    X = normal[[f"p{i}" for i in range(SEQ_LEN)]].values
    snr = normal["snr"].values.reshape(-1,1)
    # include snr as an extra input feature appended to sequence across time: replicate snr across time
    snr_rep = np.repeat(snr, SEQ_LEN, axis=1)  # shape (n, SEQ_LEN)
    X_full = np.stack([X, snr_rep], axis=-1)  # (n, SEQ_LEN, 2)
    # standardize per-feature (flatten by time)
    n, t, f = X_full.shape
    X_flat = X_full.reshape(n*t, f)
    scaler = StandardScaler().fit(X_flat)
    X_scaled = scaler.transform(X_flat).reshape(n, t, f)

    # split
    X_tr, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

    inp = Input(shape=(SEQ_LEN, 2))
    # encoder
    x = GRU(64, return_sequences=True)(inp)
    x = GRU(32, return_sequences=False)(x)
    # bottleneck
    encoded = Dense(16, activation="relu")(x)
    # decoder — expand back to sequence
    x = Dense(32, activation="relu")(encoded)
    x = Dense(SEQ_LEN*2, activation="linear")(x)
    decoded = tf.reshape(x, (-1, SEQ_LEN, 2))

    ae = Model(inputs=inp, outputs=decoded)
    ae.compile(optimizer="adam", loss="mse")
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    ae.fit(X_tr, X_tr, validation_data=(X_val, X_val), epochs=50, batch_size=256, callbacks=callbacks, verbose=1)

    # Save scaler + AE
    ae.save(os.path.join(MODEL_DIR, "gru_ae.h5"))
    import joblib
    joblib.dump(scaler, os.path.join(MODEL_DIR, "gru_ae_scaler.pkl"))
    print("Saved GRU-AE and scaler.")

    # Determine reconstruction error threshold using a small mixed validation (normal + small faults)
    # load some faults for threshold tuning
    faults = pd.read_csv(os.path.join(DATA_DIR, "faults.csv"))
    Xf = faults[[f"p{i}" for i in range(SEQ_LEN)]].values
    snrf = faults["snr"].values.reshape(-1,1)
    snrf_rep = np.repeat(snrf, SEQ_LEN, axis=1)
    Xf_full = np.stack([Xf, snrf_rep], axis=-1)
    Xf_flat = Xf_full.reshape(Xf_full.shape[0]*SEQ_LEN, 2)
    Xf_scaled = scaler.transform(Xf_flat).reshape(Xf_full.shape[0], SEQ_LEN, 2)

    # compute reconstruction errors
    recon_norm = ae.predict(X_val, batch_size=256)
    err_norm = np.mean(np.square(recon_norm - X_val), axis=(1,2))
    recon_fault = ae.predict(Xf_scaled[:2000], batch_size=256)
    err_fault = np.mean(np.square(recon_fault - Xf_scaled[:2000]), axis=(1,2))
    # choose threshold maximizing F1 (simple grid)
    best_thr = 0.0
    best_f1 = 0.0
    from sklearn.metrics import precision_recall_fscore_support
    combined_err = np.concatenate([err_norm, err_fault])
    labels = np.concatenate([np.zeros_like(err_norm), np.ones_like(err_fault)])  # 0=normal,1=fault
    for thr in np.linspace(np.min(combined_err), np.max(combined_err), 100):
        preds = (combined_err > thr).astype(int)
        p,r,f,_ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = thr
    # save threshold
    import json
    json.dump({"threshold": float(best_thr)}, open(os.path.join(MODEL_DIR, "gru_ae_threshold.json"), "w"))
    print("GRU-AE threshold set to", best_thr, "best F1", best_f1)

# -------------------------
# 2) Train Attention-BiGRU multitask (diagnose + localize)
# -------------------------
def train_attention_bigrul():
    print("Loading faults for BiGRU training...")
    faults = pd.read_csv(os.path.join(DATA_DIR, "faults.csv"))
    # features
    X = faults[[f"p{i}" for i in range(SEQ_LEN)]].values
    snr = faults["snr"].values.reshape(-1,1)
    # append snr as a feature channel
    snr_rep = np.repeat(snr, SEQ_LEN, axis=1)
    X_full = np.stack([X, snr_rep], axis=-1)  # (n,30,2)
    # standardize
    n,t,f = X_full.shape
    X_flat = X_full.reshape(n*t, f)
    scaler = StandardScaler().fit(X_flat)
    Xs = scaler.transform(X_flat).reshape(n,t,f)
    # labels: fault_type -> encoder, position -> regression
    y_cls = LabelEncoder().fit_transform(faults["fault_type"].values)
    y_pos = faults["position"].values.astype(np.float32)

    X_train, X_test, y_cls_tr, y_cls_te, y_pos_tr, y_pos_te = train_test_split(
        Xs, y_cls, y_pos, test_size=0.2, random_state=42, stratify=y_cls)

    num_classes = len(np.unique(y_cls))
    # model
    inp = Input(shape=(SEQ_LEN, 2))
    x = Bidirectional(GRU(64, return_sequences=True))(inp)
    x = Bidirectional(GRU(32, return_sequences=True))(x)
    # attention
    att = AttentionLayer()(x)  # (batch, features)
    shared = Dense(64, activation="relu")(att)
    shared = Dropout(0.3)(shared)
    # Task 1: classification
    cls_out = Dense(32, activation="relu")(shared)
    cls_out = Dense(num_classes, activation="softmax", name="cls_out")(cls_out)
    # Task 2: localization regression
    loc_out = Dense(32, activation="relu")(shared)
    loc_out = Dense(1, activation="linear", name="loc_out")(loc_out)

    model = Model(inputs=inp, outputs=[cls_out, loc_out])
    # Weighted loss
    lambda1 = 1.0  # classification weight
    lambda2 = 1.0  # localization weight; tune if needed
    losses = {"cls_out":"sparse_categorical_crossentropy", "loc_out":"mse"}
    loss_weights = {"cls_out":lambda1, "loc_out":lambda2}
    model.compile(optimizer="adam", loss=losses, loss_weights=loss_weights, metrics={"cls_out":"accuracy"})
    model.summary()

    ckpt = ModelCheckpoint(os.path.join(MODEL_DIR, "abigrul_best.h5"), save_best_only=True,
                           monitor="val_cls_out_accuracy", mode="max", verbose=1)
    es = EarlyStopping(monitor="val_cls_out_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, [y_cls_tr, y_pos_tr],
              validation_split=0.2, epochs=50, batch_size=256, callbacks=[ckpt, es], verbose=1)

    model.save(os.path.join(MODEL_DIR, "abigrul_final.h5"))
    import joblib
    joblib.dump(scaler, os.path.join(MODEL_DIR, "abigrul_scaler.pkl"))
    print("Saved Attention-BiGRU model and scaler.")

if __name__ == "__main__":
    train_gru_ae()
    train_attention_bigrul()
