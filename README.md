# NIDS-ML  
🚨 Network Intrusion Detection System using Machine Learning

This project implements a **Network Intrusion Detection System (NIDS)** using Machine Learning techniques to classify network traffic as **normal** or **malicious**.  
It follows an **end-to-end ML pipeline** approach — from data ingestion to model training, evaluation, and prediction.

The focus of this project is not just accuracy, but **clean architecture, reproducibility, and real-world ML workflow**.

---

## 📌 Problem Statement

Traditional rule-based intrusion detection systems struggle with:
- New and evolving attack patterns
- High false-positive rates
- Scalability issues

This project uses **Machine Learning models** to learn patterns from historical network traffic data and automatically detect intrusions.

---

## 🧠 Solution Overview

The system:
1. Ingests network traffic data
2. Cleans and preprocesses features
3. Trains ML models to detect intrusions
4. Evaluates performance using standard metrics
5. Uses a prediction pipeline for unseen data

The project structure is designed to be **modular, extendable, and production-ready**.

---

## ✨ Key Features

- End-to-end ML pipeline architecture  
- Data ingestion, transformation, and model training modules  
- Input validation for prediction pipeline  
- Model persistence using pickle  
- Clean separation of concerns (components, pipeline, utils)  
- Git-versioned and reproducible  

---

## 🗂️ Project Structure
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
│   │   └── predict_pipeline.py
│   │
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
│
├── artifacts/               
├── requirements.txt
├── setup.py
└── README.md


## ⚙️ Tech Stack

- **Programming Language:** Python  
- **Libraries:**  
  - NumPy, Pandas  
  - Scikit-learn  
  - Pickle  
- **Tools:**  
  - Git & GitHub  
  - VS Code  

---

## 📊 Dataset

The model is trained on a **network intrusion dataset** (e.g., NSL-KDD or similar structured traffic data).

Features typically include:
- Protocol information  
- Service type  
- Network behavior metrics  

> The dataset is split into training and testing sets for unbiased evaluation.

---
