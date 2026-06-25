# AI-driven-Anomaly-Detection-System-using-Hybrid-Machine-Learning-for-Large-Scale-Network-Datastreams
A real-time Network Intrusion Detection System (NIDS) that combines machine learning and network packet analysis to detect malicious activities in network traffic. The system captures live packets using Scapy, performs anomaly detection and attack classification using a hybrid ML pipeline, and visualizes results through an interactive Python dashboard.

---

## Project Overview

This project was developed to detect network intrusions in real time by integrating:

- Live packet capture
- Feature extraction
- Anomaly detection
- Attack classification
- Real-time monitoring dashboard

The system uses both unsupervised and supervised machine learning techniques to identify known and unknown network attacks while minimizing false alarms.

---

## Features

- Real-time packet capture using Scapy
- Network traffic feature extraction
- Hybrid Machine Learning Detection Framework
  - Isolation Forest for anomaly detection
  - Autoencoder for anomaly detection
  - XGBoost for attack classification
- Real-time threat monitoring dashboard
- Detection of both known and unknown attacks
- Alert generation for suspicious activities
- Performance visualization

---

## System Architecture

```text
Network Traffic
       │
       ▼
 Packet Capture (Scapy)
       │
       ▼
 Feature Extraction
       │
       ▼
 ┌─────────────────────┐
 │ Hybrid ML Engine    │
 ├─────────────────────┤
 │ Isolation Forest    │
 │ Autoencoder         │
 │ XGBoost             │
 └─────────────────────┘
       │
       ▼
 Threat Detection
       │
       ▼
 Real-Time Dashboard
```

---

## Machine Learning Models

### Isolation Forest
Used for unsupervised anomaly detection to identify suspicious network behavior that deviates from normal traffic patterns.

### Autoencoder
Learns normal traffic characteristics and flags packets with high reconstruction error as anomalies.

### XGBoost
Performs supervised classification of known attack categories based on extracted network features.

---

## Technologies Used

### Programming Languages
- Python

### Libraries & Frameworks
- Scapy
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Streamlit / FastAPI

### Dataset
- CICIDS2017 Dataset

---

## Project Structure

```text
NIDS/
│
├── README.md
├── mini_final.ipynb          # Model training and experimentation
├── nids.py                   # Main application
├── requirements.txt
│
├── models/
│   ├── isolation_forest.pkl
│   ├── autoencoder.h5
│   └── xgboost_model.pkl
│
├── dashboard/
│   └── dashboard.py
│
└── dataset/
    └── CICIDS2017.csv
```
---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/network-intrusion-detection-system.git

cd network-intrusion-detection-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train Models

Open and run:

```bash
mini_final.ipynb
```

### Start Real-Time Detection

```bash
python nids.py
```

### Launch Dashboard

```bash
python dashboard.py
```

---

## Performance

| Metric | Score |
|----------|----------|
| Accuracy | ~98% |
| Recall | ~96% |
| Precision | High |
| False Alarm Rate | Low |

The hybrid detection approach enables the system to identify both previously known attacks and unseen anomalous traffic patterns.

---

## Applications

- Enterprise Network Monitoring
- Cybersecurity Research
- Security Operations Centers (SOC)
- Academic Research
- Real-Time Threat Detection

---

## Future Enhancements

- Deep Learning-based traffic classification
- Integration with SIEM platforms
- Automated response mechanisms
- Cloud deployment
- Distributed intrusion detection
- Threat intelligence integration

---

## Author

**[V. Nyshada, N. Harshitha, P. Sriram]**

B.Tech Computer Science Engineering

Interests:
- Cybersecurity
- Machine Learning
- Network Security
- Artificial Intelligence

---

## License

This project is developed for academic and research purposes.
