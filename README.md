# NIDS-ML

🚨 Network Intrusion Detection System using Machine Learning

This project implements a **Network Intrusion Detection System (NIDS)** using Machine Learning to classify network traffic as **normal or malicious**.

It follows a **complete end-to-end ML pipeline**, focusing not only on performance but also on:

* Clean architecture
* Reproducibility
* Real-world ML practices
* Explainability

---

## 📌 Problem Statement

Traditional rule-based intrusion detection systems struggle with:

* New and evolving attack patterns
* High false-positive rates
* Scalability issues

This project leverages **Machine Learning models** to automatically learn patterns and detect intrusions more effectively.

---

## 🧠 Solution Overview

The system pipeline includes:

1. **Data Ingestion** → Load and prepare raw network data
2. **Data Transformation** → Feature cleaning, scaling, handling missing values
3. **Model Training** → Train multiple ML models (RF, XGBoost, CatBoost, Ensemble)
4. **Evaluation** → Generate classification reports, confusion matrix, ROC curves
5. **Prediction Pipeline** → Inference on unseen data
6. **Explainability (XAI)** → Interpret model decisions using SHAP & LIME

---

## ✨ Key Features

* ✅ End-to-end ML pipeline architecture
* ✅ Data leakage prevention (proper train/test transformation)
* ✅ Multi-class classification support
* ✅ Label encoding with persistence
* ✅ Feature alignment across training and inference
* ✅ Evaluation pipeline with:

  * F1-score, Precision, Recall
  * Confusion Matrix
  * ROC-AUC curves
* ✅ Explainable AI:

  * SHAP (global feature importance)
  * LIME (local explanations)
* ✅ Clean modular design (components, pipeline, utils)
* ✅ Git-versioned and reproducible

---

## 📁 Project Structure

```text
NIDS-ML/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   ├── predict_pipeline.py
│   │   ├── evaluation.py
│   │   └── Xai_pipeline.py
│   │
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
│
├── models/      # ignored (generated)
├── reports/     # ignored (generated)
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Tech Stack

* **Language:** Python
* **Libraries:**

  * NumPy, Pandas
  * Scikit-learn
  * XGBoost, CatBoost
  * SHAP, LIME
  * Matplotlib, Seaborn
* **Tools:**

  * Git & GitHub
  * VS Code

---

## 🚀 How to Run

### Train Model

```bash
python -m src.pipeline.train_pipeline
```

### Evaluate Model

```bash
python -m src.pipeline.evaluation
```

### Run Explainability (XAI)

```bash
python -m src.pipeline.Xai_pipeline
```

### Predict on New Data

```bash
python -m src.pipeline.predict_pipeline
```

---

## 📊 Outputs

Generated in `reports/`:

* Classification report (CSV + PNG)
* Confusion matrix
* ROC curves
* SHAP plots

---

## 🧠 Key ML Practices Implemented

* ✔ No data leakage
* ✔ Consistent preprocessing across pipelines
* ✔ Proper encoding for training & inference
* ✔ Robust handling of missing & infinite values

---

## 🚀 Future Improvements

* Deploy using Flask / FastAPI
* Real-time network monitoring
* Model versioning & tracking

---

## 👤 Author

**Hetvi Kakkad**
