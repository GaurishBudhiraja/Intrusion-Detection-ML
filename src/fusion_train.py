# src/fusion_train.py
import os, json, numpy as np, pandas as pd, joblib, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Input, GRU, Bidirectional, Dense, Dropout, Concatenate, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN = 30
NET_DIM = 10

class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True, name="W_att")
        self.u = self.add_weight(shape=(input_shape[-1],1),
                                 initializer="glorot_uniform", trainable=True, name="u_att")
        super().build(input_shape)
    def call(self, inputs):
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1))
        ait = tf.nn.softmax(tf.tensordot(uit, self.u, axes=1), axis=1)
        return tf.reduce_sum(inputs * ait, axis=1)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # optical tensor (batch, 30, 2) -> (p0..p29, snr replicated)
    Xp = df[[f"p{i}" for i in range(SEQ_LEN)]].values.astype(np.float32)
    snr = df["snr"].values.astype(np.float32).reshape(-1,1)
    snr_rep = np.repeat(snr, SEQ_LEN, axis=1)
    X_opt = np.stack([Xp, snr_rep], axis=-1)

    X_net = df[[f"net_{i}" for i in range(NET_DIM)]].values.astype(np.float32)

    y_label = df["label"].values
    y_pos = df["position"].values.astype(np.float32)

    # scalers
    n,t,c = X_opt.shape
    opt_scaler = StandardScaler().fit(X_opt.reshape(n*t*c//c, c))
    X_opt_s = opt_scaler.transform(X_opt.reshape(-1, c)).reshape(n,t,c)

    net_scaler = StandardScaler().fit(X_net)
    X_net_s = net_scaler.transform(X_net)

    le = LabelEncoder().fit(np.concatenate([y_label, ["normal"]]))
    y = le.transform(y_label)

    return (X_opt_s, X_net_s, y, y_pos, le, opt_scaler, net_scaler)

def build_model(num_classes):
    # optical branch
    inp_opt = Input(shape=(SEQ_LEN, 2), name="opt_input")
    x = Bidirectional(GRU(64, return_sequences=True))(inp_opt)
    x = Bidirectional(GRU(32, return_sequences=True))(x)
    x = AttentionLayer()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    # network branch
    inp_net = Input(shape=(NET_DIM,), name="net_input")
    n = Dense(64, activation="relu")(inp_net)
    n = Dropout(0.2)(n)
    n = Dense(32, activation="relu")(n)

    fused = Concatenate()([x, n])
    fused = Dense(128, activation="relu")(fused)

    cls_out = Dense(num_classes, activation="softmax", name="cls_out")(fused)
    loc_out = Dense(1, activation="linear", name="loc_out")(fused)  # for physical attacks

    model = Model(inputs=[inp_opt, inp_net], outputs=[cls_out, loc_out])
    model.compile(
        optimizer="adam",
        loss={"cls_out":"sparse_categorical_crossentropy", "loc_out":"mse"},
        loss_weights={"cls_out":1.0, "loc_out":1.0},
        metrics={"cls_out":"accuracy"}
    )
    return model

def train(csv="hybrid.csv"):
    X_opt, X_net, y, y_pos, le, opt_scaler, net_scaler = load_data(os.path.join(DATA_DIR, csv))
    Xo_tr, Xo_te, Xn_tr, Xn_te, y_tr, y_te, pos_tr, pos_te = train_test_split(
        X_opt, X_net, y, y_pos, test_size=0.2, random_state=42, stratify=y)

    model = build_model(num_classes=len(np.unique(y)))
    ckpt = ModelCheckpoint(os.path.join(MODEL_DIR, "fusion_best.h5"),
                           monitor="val_cls_out_accuracy", mode="max",
                           save_best_only=True, verbose=1)
    es = EarlyStopping(monitor="val_cls_out_accuracy", patience=6,
                       mode="max", restore_best_weights=True)

    model.fit(
        [Xo_tr, Xn_tr], [y_tr, pos_tr],
        validation_split=0.2,
        epochs=40, batch_size=256,
        callbacks=[ckpt, es], verbose=1
    )

    model.save(os.path.join(MODEL_DIR, "fusion_final.h5"))
    joblib.dump(opt_scaler, os.path.join(MODEL_DIR, "fusion_opt_scaler.pkl"))
    joblib.dump(net_scaler, os.path.join(MODEL_DIR, "fusion_net_scaler.pkl"))
    json.dump({"classes": le.classes_.tolist()}, open(os.path.join(MODEL_DIR, "fusion_labels.json"), "w"))
    print("Saved fusion model + scalers + label map.")

if __name__ == "__main__":
    train()
