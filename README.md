# 🔐 Intrusion Detection in Submarine Fiber-Optic Cables (Hybrid AI)

This repository implements a **hybrid AI-based Intrusion Detection System (IDS)** designed for **submarine fiber-optic cables**.  
The system combines **optical-layer metrics** (signal quality, BER, power leakage) with **network traffic features** (flow entropy, TLS handshake patterns) to detect **both cyber and physical attacks** in real time.  

---

## ✨ Highlights
- **Hybrid Model:** Combines 1D-CNN, Bi-LSTM with Attention, Autoencoders, and Bi-GRU.  
- **Dual-Layer Security:** Detects anomalies at both **physical** and **network layers**.  
- **Performance:** Achieves **~97% accuracy** with very low false positives (~3%).  
- **Attacks Detected:** Fiber bending, covert modulation, SNR degradation, tap insertion, MITM.  

---

## 📂 Repository Structure
├── datasets/ # Hybrid datasets (optical + network features)
│ ├── optical_signals.csv # Physical-layer features (SNR, bending, etc.)
│ ├── network_traffic.csv # Network-layer features (packets, flows)
│ └── hybrid_dataset.csv # Combined dataset (fusion)
│
├── models/ # Pretrained and saved models
│ ├── abigrul_best.h5 # Bi-GRU + Attention model
│ ├── fusion_best.h5 # Hybrid fusion model
│ ├── fusion_labels.json # Class labels
│ ├── *_scaler.pkl # Scalers for normalization
│ └── ...
│
├── src/ # Source code
│ ├── baselines.py # Traditional IDS baselines
│ ├── train_model.py # Train individual models
│ ├── evaluate.py # Evaluate single models
│ ├── fusion_train.py # Train hybrid fusion model
│ ├── evaluate_hybrid.py # Evaluate hybrid model
│ ├── gen_datasets.py # Dataset generation & preprocessing
│ ├── hybrid_gen.py # Hybrid dataset creator
│ ├── visualize.py # Visualizations & plots
│ ├── infer.py # Inference on new samples
│ └── run_all.py # Run complete pipeline
│
└── README.md

---

This way, both **hybrid datasets** and **hybrid model scripts** are clearly shown.  

Do you also want me to include a **📊 Workflow Diagram** (ASCII-style in the README) to show how **optical + network features → hybrid model → intrusion detection**?


## 🚀 Usage

### Clone the repo
```bash
git clone https://github.com/GaurishBudhiraja/Intrusion-Detection-ML.git
cd Intrusion-Detection-ML
```

📊 Results (Summary)
      Method	                Accuracy	  F1-score	False Positive Rate
Rule-based IDS (Snort)	         53%	      36%	            2%
Threshold IDS	                   60%	      75%	            39%
Hybrid AI (ours)	               97%	      97%	            3%

🔮 Future Work

Real-world optical-layer dataset integration.

Adversarial ML defense.

Integration with SDN/SOAR controllers.

Explainable AI (XAI) for operator-friendly detection.
