# src/hybrid_gen.py
import os, numpy as np, pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "datasets")
os.makedirs(DATA_DIR, exist_ok=True)

SEQ_LEN = 30
RNG = np.random.default_rng(7)

ATTACKS = [
    "tap_insertion",        # physical
    "fiber_bending",        # physical
    "covert_modulation",    # physical
    "low_rate_snr_loss",    # physical/network crossover
    "network_mitm",         # network
    "packet_injection"      # network
]

def _optical_trace(label, snr):
    """Return (seq[30], snr)."""
    # baseline attenuating shape with small ripple
    base = np.linspace(0.1, 0.01, SEQ_LEN) + RNG.normal(0, 0.0015, SEQ_LEN)
    seq = base.copy()

    # inject label-specific signatures
    pos = RNG.integers(5, SEQ_LEN-5)
    if label == "tap_insertion":
        seq[pos:pos+3] -= RNG.uniform(0.01, 0.04)
    elif label == "fiber_bending":
        seq[pos-1:pos+2] += RNG.uniform(0.015, 0.06)
    elif label == "covert_modulation":
        w = RNG.uniform(2.0, 3.5)
        seq += 0.006*np.sin(np.linspace(0, w*np.pi, SEQ_LEN))
    elif label == "low_rate_snr_loss":
        # gentle, global degradation
        seq += np.linspace(0, -0.03, SEQ_LEN)
    elif label in ("network_mitm", "packet_injection"):
        # mostly clean optical but keep background noise
        pass

    # noise scales inversely with snr
    noise_scale = float(np.clip(0.05/(snr+1e-6), 0.0005, 0.02))
    seq += RNG.normal(0, noise_scale, SEQ_LEN)
    return seq

def _network_features(label):
    """Return a small vector of network features that shifts by label."""
    # features: [mean_pkt, std_pkt, mean_iat, std_iat, flow_dur, syn_ratio,
    #            fin_ratio, bytes_per_flow, pkt_entropy_proxy, iat_entropy_proxy]
    mean_pkt = RNG.normal(600, 80)      # bytes
    std_pkt  = abs(RNG.normal(120, 30))
    mean_iat = abs(RNG.normal(12, 3))   # ms
    std_iat  = abs(RNG.normal(6, 2))
    flow_dur = abs(RNG.normal(180, 50)) # ms
    syn_ratio= RNG.uniform(0.02, 0.08)
    fin_ratio= RNG.uniform(0.02, 0.08)
    bpf      = abs(RNG.normal(2.2e4, 5e3))
    pe       = RNG.uniform(2.2, 3.0)
    ie       = RNG.uniform(2.2, 3.0)

    if label == "network_mitm":
        mean_iat *= 1.35; std_iat *= 1.4; syn_ratio *= 1.6; pe *= 1.2
    if label == "packet_injection":
        mean_pkt *= 0.8; std_pkt *= 1.6; syn_ratio *= 2.1; bpf *= 1.3; ie *= 1.25
    if label == "covert_modulation":
        pe *= 1.15; ie *= 1.15
    if label == "low_rate_snr_loss":
        mean_iat *= 1.1; std_iat *= 1.1
    if label == "tap_insertion":
        # slight effect on bytes/flow due to small losses/retransmits
        bpf *= 0.95
    if label == "fiber_bending":
        bpf *= 0.9

    return np.array([mean_pkt, std_pkt, mean_iat, std_iat, flow_dur,
                     syn_ratio, fin_ratio, bpf, pe, ie], dtype=np.float32)

def make_hybrid(n_normal=60000, n_attack_each=15000, out_prefix="hybrid"):
    rows = []
    # Normal
    for _ in range(n_normal):
        snr = RNG.uniform(15, 35)
        seq = _optical_trace("normal", snr)
        net = _network_features("normal")
        rows.append(np.concatenate([seq, [snr], net, ["normal", -1]]))

    # Attacks
    for label in ATTACKS:
        for _ in range(n_attack_each):
            snr = RNG.uniform(3, 25) if label != "network_mitm" else RNG.uniform(8, 28)
            seq = _optical_trace(label, snr)
            net = _network_features(label)
            pos = int(RNG.integers(5, SEQ_LEN-5)) if "network" not in label else -1
            rows.append(np.concatenate([seq, [snr], net, [label, pos]]))

    cols = [f"p{i}" for i in range(SEQ_LEN)] + ["snr"] + \
           [f"net_{i}" for i in range(10)] + ["label", "position"]
    df = pd.DataFrame(rows, columns=cols)
    df["position"] = df["position"].astype(int)
    df.to_csv(os.path.join(DATA_DIR, f"{out_prefix}.csv"), index=False)
    print("Saved:", os.path.join(DATA_DIR, f"{out_prefix}.csv"), df.shape)

if __name__ == "__main__":
    make_hybrid()
