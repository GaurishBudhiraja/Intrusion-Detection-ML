# gen_datasets.py
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
DATA_DIR = os.path.join(BASE_DIR, "..", "datasets")
os.makedirs(DATA_DIR, exist_ok=True)

SEQ_LEN = 30

def make_normal(n_samples=48000):
    rows = []
    for i in range(n_samples):
        # create smooth baseline OTDR-like trace + small noise
        base = np.linspace(0.1, 0.01, SEQ_LEN)  # decreasing attenuation shape (example)
        noise = np.random.normal(scale=0.002, size=SEQ_LEN)
        seq = base + noise + np.random.normal(scale=0.001, size=SEQ_LEN)
        snr = np.random.uniform(2.0, 30.0)  # SNR for the sequence
        row = np.concatenate([seq, [snr]])
        rows.append(row)
    cols = [f"p{i}" for i in range(SEQ_LEN)] + ["snr"]
    df = pd.DataFrame(rows, columns=cols)
    df["label"] = "normal"
    df.to_csv(os.path.join(DATA_DIR, "gru_ae_normal.csv"), index=False)
    print("Saved gru_ae_normal.csv:", df.shape)

def make_faults(n_samples=61849):
    fault_types = ["fiber_tapping","bad_splice","dirty_connector","fiber_cut"]
    rows = []
    for i in range(n_samples):
        base = np.linspace(0.1, 0.01, SEQ_LEN)
        snr = np.random.uniform(0.0, 30.0)
        ftype = np.random.choice(fault_types, p=[0.25,0.25,0.25,0.25])
        # inject fault pattern at random position
        pos = np.random.randint(5, SEQ_LEN-5)
        seq = base + np.random.normal(scale=0.002, size=SEQ_LEN)
        if ftype == "fiber_tapping":
            # slight dip near pos (evanescent coupling)
            seq[pos:pos+3] -= np.random.uniform(0.01, 0.05)
        elif ftype == "bad_splice":
            seq[pos:pos+2] += np.random.uniform(0.02, 0.1)
        elif ftype == "dirty_connector":
            seq[pos] += np.random.uniform(0.005, 0.03)
        elif ftype == "fiber_cut":
            # large drop after pos
            seq[pos:] = seq[pos] + np.linspace(-0.05, -0.2, SEQ_LEN-pos)
        # add more noise for lower SNR
        noise_scale = np.clip(0.05 / (snr+1e-6), 0.0005, 0.02)
        seq += np.random.normal(scale=noise_scale, size=SEQ_LEN)
        row = np.concatenate([seq, [snr]])
        rows.append(np.concatenate([row, [ftype, pos]]))
    cols = [f"p{i}" for i in range(SEQ_LEN)] + ["snr","fault_type","position"]
    df = pd.DataFrame(rows, columns=cols)
    # position saved as float from concatenation; cast to int
    df["position"] = df["position"].astype(int)
    df.to_csv(os.path.join(DATA_DIR, "faults.csv"), index=False)
    print("Saved faults.csv:", df.shape)

if __name__ == "__main__":
    print("Generating datasets in:", DATA_DIR)
    make_normal(n_samples=47904)   # match paper-ish sizes if you want
    make_faults(n_samples=61849)
