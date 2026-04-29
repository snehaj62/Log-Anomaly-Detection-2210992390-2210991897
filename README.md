# Log Anomaly Detection: Rule-Based vs Machine Learning


**Comparative Evaluation of Rule-Based and Lightweight Machine Learning Approaches for Log Anomaly Detection**

*COOP-2 Project Report | Chitkara University*


---

## Authors

| Name | Roll No. | Email |
|------|----------|-------|
| Manya Bajaj | 2210991897 | manya1897.be22@chitkara.edu.in |
| Sneha Jindal | 2210992390 | sneha2390.be22@chitkara.edu.in |

**Supervisor:** Dr. Rajat Takkar — Department of CSE, Chitkara University

---

##  Abstract

Modern distributed systems generate massive volumes of logs that must be continuously monitored to detect failures and security threats. This project implements and compares **three anomaly detection approaches**:

-  **Rule-Based Detection** — predefined thresholds & keyword matching
-  **Logistic Regression** — supervised binary classification
-  **Isolation Forest** — unsupervised anomaly isolation

The study evaluates each method on **Precision, Recall, F1-Score, False Positive Rate, and Detection Latency** using a hybrid dataset of real and synthetic system logs.

---

## Key Findings

| Model | Strength | Weakness |
|-------|----------|----------|
| Rule-Based | Fast, zero training needed | High false positives, not adaptive |
| Logistic Regression | High precision on known patterns | Struggles with unseen anomaly types |
| Isolation Forest | Best for novel/rare anomalies | Slightly higher latency |

>  **Conclusion:** A **hybrid approach** combining rule-based speed with ML adaptability is most effective for production systems.

---

## Project Structure

```
log-anomaly-detection/
│
├── data/
│   └── generate_dataset.py       # Synthetic log dataset generator
│
├── models/
│   ├── rule_based.py             # Rule-based detector (keywords + thresholds)
│   ├── logistic_regression.py    # Supervised ML classifier
│   └── isolation_forest.py       # Unsupervised anomaly isolator
│
├── utils/
│   ├── preprocess.py             # TF-IDF feature extraction & encoding
│   └── evaluate.py               # Metrics computation & reporting
│
├── results/
│   └── comparison_report.txt     # Auto-generated evaluation report
│
├── main.py                       # ▶ Run full pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/log-anomaly-detection.git
cd log-anomaly-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python main.py
```

---

## How It Works

```
System Logs
    │
    ▼
[1] Data Generation      → 2000 synthetic logs (5% anomalies injected)
    │
    ▼
[2] Preprocessing        → Clean text, TF-IDF vectorize, encode log level & service
    │
    ▼
[3] Feature Extraction   → 500 TF-IDF features + categorical encodings
    │
    ├──▶ Rule-Based Model      → Keyword match + log level + sliding window
    ├──▶ Logistic Regression   → Trained on labeled feature vectors
    └──▶ Isolation Forest      → Unsupervised, no labels required
    │
    ▼
[4] Evaluation           → Precision, Recall, F1, FPR, Latency
    │
    ▼
[5] Comparison Report    → Saved to results/comparison_report.txt
```

---

## Evaluation Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Precision** | TP / (TP + FP) | How many flagged anomalies are real |
| **Recall** | TP / (TP + FN) | How many real anomalies are caught |
| **F1-Score** | 2 × (P × R) / (P + R) | Balance between Precision & Recall |
| **FPR** | FP / (FP + TN) | Rate of false alarms on normal logs |
| **Latency** | Wall-clock seconds | Speed of detection on test set |

---

##  Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| scikit-learn | ML models (LR, Isolation Forest) + TF-IDF |
| pandas | Dataset handling & preprocessing |
| NumPy | Numerical operations & feature matrix |

---

## References

1. S. Hussein and S. Répás, *"Anomaly detection in log files based on machine learning techniques"*, Journal of Electrical Systems, 2024.
2. Y. Lee, J. Kim, and P. Kang, *"LAnoBERT: System log anomaly detection based on BERT masked language model"*, 2021.
3. Y. Xie, H. Zhang, and M. A. Babar, *"LogGD: Detecting anomalies from system logs using graph neural networks"*, 2022.
4. B. Du et al., *"DeepLog: Anomaly detection and diagnosis from system logs through deep learning"*, IEEE INFOCOM.
5. J. Yang et al., *"Improved principal component analysis for log anomaly detection"*, 2023.
6. Q. Zhang et al., *"Log anomaly detection using contrastive learning and retrieval augmentation"*, Scientific Reports, 2025.

---

## Institution

**Chitkara University Institute of Engineering and Technology**
Department of Computer Science and Engineering
Chitkara University, Punjab, India
