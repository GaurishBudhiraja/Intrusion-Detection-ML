# src/run_all.py
import os, json, pandas as pd
from hybrid_gen import make_hybrid
from fusion_train import train
from evaluate_hybrid import eval_fusion, eval_baselines

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

if __name__ == "__main__":
    print(">>> Generating hybrid dataset …")
    make_hybrid(n_normal=60000, n_attack_each=15000, out_prefix="hybrid")  # ~150k total

    print("\n>>> Training fusion model …")
    train(csv="hybrid.csv")

    print("\n>>> Evaluating …")
    eval_fusion()
    eval_baselines()
    print("\nArtifacts saved in:", MODEL_DIR)
