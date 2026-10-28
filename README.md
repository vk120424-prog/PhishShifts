[README.md](https://github.com/user-attachments/files/34087621/README.md)
# 🛡️ PhishShift – Message Risk Scanner

An advanced, explainable cybersecurity desktop and web application designed to analyze email and SMS messages for phishing threats, social engineering patterns, shortened URLs, and deceptive links—without opening suspicious links or exposing users to malicious payloads.

Built as an academic and educational cybersecurity college project demonstrating machine learning text classification, transparent heuristic risk scoring, safe offline URL analysis, and modern security dashboard visualization.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Key Features](#key-features)
5. [Technology Stack](#technology-stack)
6. [Project Structure](#project-structure)
7. [Dataset Format & Provenance](#dataset-format--provenance)
8. [Installation & Setup (Windows PowerShell / VS Code)](#installation--setup-windows-powershell--vs-code)
9. [Training the Machine Learning Model](#training-the-machine-learning-model)
10. [Launching the Web Application](#launching-the-web-application)
11. [Running Automated Tests & Acceptance Criteria](#running-automated-tests--acceptance-criteria)
12. [Security, Privacy & Educational Limitations](#security-privacy--educational-limitations)
13. [Future Enhancements](#future-enhancements)

---

## 📖 Project Overview
Phishing and SMS phishing (*smishing*) remain the leading initial access vectors for cyberattacks, credential theft, and ransomware. Everyday users and organizations often struggle to discern deceptive communications from legitimate notifications—especially when messages deploy psychological manipulation (e.g., urgency, account termination threats) and obfuscated links.

**PhishShift** provides a defensive inspection tool that evaluates text messages purely offline. It combines:
- A **Scikit-Learn Machine Learning Classifier** (TF-IDF + Balanced Logistic Regression).
- An **Explainable Rule-Based Heuristic Engine** (detecting urgency, credential requests, scam lures, and formatting anomalies).
- A **Safe Offline URL Analyzer** (detecting URL shorteners, raw IP hosts, brand imitation/typosquatting, and high-risk TLDs without pinging the web).

---

## 🎯 Problem Statement
Traditional anti-phishing tools either rely on closed-source black-box models or require clicking / querying external threat intelligence APIs, which risks leaking private email contents and alerting attackers. Users require an educational, transparent tool that:
1. Ingests raw email or SMS text safely.
2. Identifies specific structural and linguistic threat indicators.
3. Quantifies risk via a heuristic score (0–100) and three clear categories: **LOW RISK**, **MEDIUM RISK**, and **HIGH RISK**.
4. Clearly explains **why** a message was flagged with actionable defense recommendations.

---

## 🚀 Objectives
- **Safe Offline Inspection**: Never visit suspicious links, query DNS servers, or make external HTTP calls during analysis.
- **Explainable AI (XAI)**: Demystify risk classification by detailing exactly which phrases and domain attributes triggered suspicion.
- **Hybrid Scoring**: Blend statistical ML probabilities with deterministic rule penalties to prevent false negatives on deceptive new templates.
- **College Project Ready**: Clean, modern dark navy cybersecurity UI with real-time feedback, sample loaders, model performance metrics, and automated test coverage.

---

## ✨ Key Features
- **Modern Cyber Dashboard**: High-contrast dark navy UI (`#0a0f1d`) with neon cyan (`#00f0ff`) and blue accents.
- **Instant Testing Presets**: Preloaded with 8 diverse test messages (legitimate password resets, bank alerts, delivery notifications, lottery scams, crypto giveaway traps).
- **Offline URL Analysis**:
  - URL extraction (http/https/www/schemeless shorteners).
  - URL shortener identification (`bit.ly`, `tinyurl.com`, `t.co`, `is.gd`, etc.).
  - Raw IPv4 address detection (`http://192.168.1.1/...`).
  - Brand typosquatting and leetspeak heuristic detection (e.g., `paypa1`, `micros0ft`, `amaz0n`, `chase-verify`).
  - High-risk top-level domain (TLD) warnings (`.xyz`, `.top`, `.click`, `.buzz`, `.cam`).
- **Linguistic Threat Analysis**:
  - Urgency & panic language (e.g., *account suspended*, *within 24 hours*, *immediate action required*).
  - Credential harvesting requests (e.g., *verify password*, *confirm SSN*, *PIN*, *CVV*).
  - Financial lures & fraud (e.g., *lottery winner*, *wire transfer*, *bitcoin giveaway*, *IRS refund*).
  - Formatting anomalies (excessive exclamation marks, multiple question marks, high uppercase word ratio).
- **Risk Result Card & Meter**:
  - **LOW RISK (0 – 34)**: Safe or normal communication patterns.
  - **MEDIUM RISK (35 – 69)**: Elevated urgency or link obfuscation; verification recommended.
  - **HIGH RISK (70 – 100)**: Strong indicators of credential theft or fraud.
- **Model Performance Analytics**:
  - Interactive confusion matrix heatmap (Matplotlib).
  - Test accuracy, precision, recall, and F1-score metric cards.
  - Complete classification report.
- **Audit History**: In-memory session logging of analyzed messages with one-click CSV export.

---

## 🛠️ Technology Stack
- **Language**: Python 3.11+
- **Frontend / Web UI**: Streamlit
- **Machine Learning**: Scikit-learn (TF-IDF Vectorizer + Logistic Regression)
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Matplotlib
- **Model Persistence**: Joblib
- **Pattern Matching**: Regular Expressions (`re`) & Python `urllib.parse`
- **Testing**: Pytest

---

## 📂 Project Structure
```text
PhishShift/
│
├── app.py                      # Main Streamlit web application & cyber dashboard
├── train_model.py              # ML training, data validation, metrics & artifact export
├── message_analyzer.py         # Hybrid risk engine (ML + rule-based indicators)
├── url_analyzer.py             # Safe offline URL extraction and static heuristic analysis
├── requirements.txt            # Project dependencies
├── conftest.py                 # Pytest root configuration
├── pytest.ini                  # Pytest runner settings
├── README.md                   # Comprehensive project documentation
│
├── data/
│   └── messages.csv            # Labeled demonstration dataset (legitimate vs. phishing)
│
├── models/
│   ├── model.joblib            # Trained Scikit-Learn Logistic Regression model
│   ├── vectorizer.joblib       # Fitted TF-IDF text vectorizer
│   └── metrics.joblib          # Evaluation metrics, confusion matrix & reports
│
└── tests/
    ├── test_message_analyzer.py # Automated test suite & acceptance criteria (AT-01, AT-02)
    └── test_url_analyzer.py     # URL inspection & acceptance criteria (AT-03)
```

---

## 📊 Dataset Format & Provenance
The dataset is located at [`data/messages.csv`](file:///c:/Users/vk120/OneDrive/Desktop/PhishShift%20Message%20Risk%20%20Scanner/data/messages.csv).

### Schema
| Column | Type | Allowed Values | Description |
|---|---|---|---|
| `text` | string | Non-empty text | The message body (email or SMS) |
| `label` | string | `legitimate`, `phishing` | Ground truth classification |

### Provenance & Quality Measures
- **Provenance**: Curated demonstrative dataset constructed from documented public phishing patterns (anti-phishing working groups, CISA cybersecurity advisories) and verified legitimate enterprise transactional notifications (GitHub, Amazon, banking portals, IT helpdesks).
- **Validation**:
  - Strict schema checking (`text`, `label`).
  - Automatic removal of null, empty, or invalid label rows.
  - Pre-split deduplication of identical messages to prevent **train/test data leakage**.
  - Stratified 80/20 train/test split preserving class distribution.

---

## 💻 Installation & Setup (Windows PowerShell / VS Code)

Follow these exact steps in **Windows PowerShell** or the integrated **VS Code Terminal**:

### Step 1: Open Project Directory in VS Code
Open the project directory in VS Code or navigate via PowerShell:
```powershell
cd "c:\Users\vk120\OneDrive\Desktop\PhishShift Message Risk  Scanner"
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv .venv
```
*(If `python` is not in your global PATH, specify `C:\Users\vk120\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv`)*

### Step 3: Activate the Virtual Environment
```powershell
.venv\Scripts\activate
```
*(You will see `(.venv)` in your terminal prompt).*

### Step 4: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

---

## 🤖 Training the Machine Learning Model
Train the TF-IDF and Logistic Regression pipeline and compute performance metrics:
```powershell
python train_model.py
```

### Expected Output:
```text
============================================================
  PhishShift: Training Machine Learning Model
============================================================
[*] Validated dataset: 40 unique records retained (from 40 raw rows).
    - Legitimate: 20
    - Phishing: 20
[*] Artifacts successfully saved to: ...\models

--- Model Performance Summary ---
  Accuracy : 0.7000
  Precision: 0.6250
  Recall   : 1.0000
  F1-Score : 0.7692

Confusion Matrix [Legitimate, Phishing]:
[[2 3]
 [0 5]]

Classification Report:
               precision    recall  f1-score   support
  legitimate       1.00      0.40      0.57         5
    phishing       0.62      1.00      0.77         5
    accuracy                           0.70        10
============================================================
```

> **Note on Graceful Degradation**: If the model is not trained prior to launching the application, `app.py` displays a non-blocking warning and operates in full rule-based heuristic mode. Users can also train the model directly via the sidebar button in the web UI.

---

## 🌐 Launching the Web Application
Run the Streamlit dashboard:
```powershell
python -m streamlit run app.py
```
Streamlit will automatically launch your default browser at:
`http://localhost:8501`

---

## 🧪 Running Automated Tests & Acceptance Criteria

Execute the full automated test suite using `pytest`:
```powershell
python -m pytest -v
```

### Acceptance Tests Validated:
- **AT-01**: A legitimate password-reset notification (e.g., standard GitHub or banking security notification without suspicious links) is **not** automatically classified as phishing.
- **AT-02**: A message utilizing unfamiliar phishing-style language is accurately assessed using available model probabilities and heuristic indicators (urgency, credential demands).
- **AT-03**: A message containing suspicious URLs (such as IP addresses, shortened links, or brand typosquatting) displays extracted URL findings and explicitly presents the static analysis limitation disclaimer.

### Test Coverage Overview:
- Legitimate communications
- Obvious phishing communications
- Messages with shortened URLs
- Messages without URLs
- Empty and whitespace input handling
- Non-string or unexpected input
- Missing dataset file errors
- Duplicate dataset row removal
- Invalid label filtering
- Graceful fallback when model artifacts are missing

---

## 🔒 Security, Privacy & Educational Limitations

1. **Zero Permanent Storage**: Submitted message texts are processed strictly in-memory within the active session. No user inputs are written to disk or transmitted to third parties.
2. **Safe Offline URL Inspection**: The system never queries external web servers, initiates HTTP connections, or resolves DNS. This protects the analyst from drive-by downloads or triggering attacker canary tokens.
3. **Static Analysis Limitations**:
   - Static inspection cannot verify live web server content, dynamic JavaScript redirects, or whether an external site is currently benign or compromised.
   - Attackers frequently adapt templates; a Low Risk score should never replace standard cybersecurity vigilance.
4. **Educational Scope**: PhishShift is designed as an educational tool and student project, not a replacement for commercial enterprise email security gateways.

---

## 🔮 Future Enhancements
- Integration of Natural Language Processing transformer models (e.g., DistilBERT or DeBERTa) for deep semantic understanding.
- Support for uploading `.eml` and `.msg` email files with header inspection (SPF, DKIM, DMARC parsing).
- Optical Character Recognition (OCR) to detect phishing images and QR code (quishing) payloads.
- Integration of local WHOIS lookups for domain registration age heuristics.

---

### 👨‍💻 College Project Summary
- **Project Title**: PhishShift – Message Risk Scanner
- **Domain**: Cybersecurity, Machine Learning, Applied NLP, Web Development
- **Status**: Complete, Tested & Verified
