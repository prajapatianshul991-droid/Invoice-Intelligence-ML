# 📦 Vendor Invoice Intelligence Portal

An end-to-end Machine Learning project that analyzes vendor purchase and invoice data to:
1. **Predict Freight Cost** for purchase orders (Regression)
2. **Flag Risky Invoices** that need manual review (Classification)

Built on real-world inventory/purchasing data (5 linked tables — purchases, purchase prices, vendor invoices, beginning & ending inventory) and deployed as an interactive Streamlit web app.

---

## 🎯 Problem Statement

Businesses handling large volumes of vendor invoices often struggle with:
- Estimating freight/shipping costs in advance for budgeting
- Manually checking every invoice for pricing mismatches or delivery delays

This project automates both using machine learning, reducing manual effort and improving cost forecasting.

---

## 🧠 Features

### 1. Freight Cost Prediction
Predicts the expected freight cost of a purchase order based on order value, using a regression model trained and compared across Linear Regression, Decision Tree, and Random Forest.

**Best Model:** Linear Regression
- MAE: 26.13 (Random Forest) *(update with your best model's actual numbers)*
- R² Score: 95.63%

### 2. Invoice Risk Flagging
Classifies whether an invoice should be flagged for manual approval, based on:
- Mismatch between invoice amount and item-level purchase total
- Abnormal receiving delay

**Model:** Random Forest Classifier (hyperparameter-tuned with GridSearchCV)
- Accuracy: 89%
- Precision (Risky class): 96%
- Recall (Risky class): 71%

---

## 🛠️ Tech Stack
- **Language:** Python
- **Data Processing:** Pandas, SQLite3
- **Machine Learning:** Scikit-learn (Linear Regression, Decision Tree, Random Forest, GridSearchCV)
- **Deployment:** Streamlit
- **Version Control:** Git & GitHub

---

## 📂 Project Structure
