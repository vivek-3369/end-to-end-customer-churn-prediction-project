# 📊 Telecom Customer Churn Prediction: End-to-End Machine Learning Pipeline

## 📝 Project Overview
Customer Churn is a critical metric for businesses, particularly in the telecommunications industry where acquiring a new customer is significantly more expensive than retaining an existing one. This project is a complete, end-to-end Machine Learning pipeline designed to predict whether a customer will discontinue their telecom services (churn) or remain loyal. By predicting churn probability, telecom companies can proactively identify at-risk customers and offer targeted retention strategies.

This project utilizes the **Telco Customer Churn dataset** from Kaggle and features a robust backend pipeline coupled with an interactive web application built with **Streamlit** and fully containerized via **Docker**.

---

## 🛠️ Technology Stack
* **Programming Language:** Python 3.12
* **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE)
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Deployment, Containerization & Web App:** Streamlit, Docker
* **Environment Management:** Conda

---

## ⚙️ Architecture & Pipeline Design
The project is built on a highly modular and scalable architecture divided into two primary pipelines:

### 1. Training Pipeline
The training pipeline automates the entire process of learning from historical data. It is constructed using the following modular components:
* **Data Ingestion:** Reads the raw data from local sources and performs a train-test split.
* **Data Transformation:** Handles data cleaning and extensive Feature Engineering. Missing values are imputed, categorical variables are encoded (One-Hot & Ordinal), and numerical features are scaled. To handle severe class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied exclusively to the training set to prevent data leakage. The entire transformation process is saved as a `preprocessor.pkl` object.
* **Model Training:** Evaluates multiple baseline algorithms including Logistic Regression, Random Forest, Decision Trees, K-Nearest Neighbors, AdaBoost, and Gradient Boosting. The best-performing model is saved as `model.pkl`.
* **Model Evaluation:** Evaluates the best model on unseen testing data. It calculates crucial classification metrics (Accuracy, Precision, Recall, ROC-AUC) and stores them in `metrics.json`.

### 2. Prediction Pipeline
The Prediction Pipeline intercepts raw user input from the front end, dynamically applies the exact feature engineering logic used during training via `preprocessor.pkl`, and passes the data to `model.pkl` to calculate the real-time probability of customer churn.

---

## 🌐 Web Application Features
The Streamlit application (`app.py`) serves as a comprehensive data dashboard featuring 4 distinct analytical tabs:

1. **🔮 Prediction Engine:** Network administrators can manually construct a customer's profile using interactive sliders and dropdowns. The model calculates the churn probability and flashes conditional warnings:
   - **Low Risk:** Probability below 30% (Success metric).
   - **Moderate Risk:** Probability between 30% - 60% (Retention warning).
   - **High Risk:** Probability exceeding 60% (Severe critical alert).
2. **🎯 Customer Segmentation:** Performs dynamic RFM (Recency, Frequency, Monetary) Analysis, clustering customers into "Champions", "Loyal", "At Risk", and "About to Churn" segments.
3. **📈 Model Performance:** Visualizes the ML model's accuracy via an interactive Confusion Matrix and bar charts directly from the saved `metrics.json` file.
4. **🔍 Key Insights:** Highlights the most critical features driving churn discovered during the EDA phase.

---

## 💼 Business Value & Strategic Recommendations
Integrating this predictive model into a telecom company's daily operations delivers immense business value through Cost Reduction and Resource Optimization.

1. **Targeted First-Year Interventions:** Since month-to-month, low-tenure customers churn the most, the business should offer heavy discounts or free premium upgrades to customer migrating them to 1-year contracts at the 6-month mark.
2. **Promote Automatic Payments:** Establish cash-back incentives or loyalty points for customers who switch from Electronic Checks to automatic Credit Card or Bank Transfer payments to create billing friction for leaving.
3. **Bundle "Sticky" Services:** Customers with Tech Support and Online Security rarely leave. Marketing should push customized bundles offering these services for a reduced price to high-risk customers.

---

## 🚀 How to Run the Project

You can run this project locally using Python or instantly via Docker!

### Option A: Run via Docker (Recommended)
You do not need to install Python or set up environments. Install Docker and pull the image directly from Docker Hub!

```bash
# 1. Pull the image from Docker Hub
docker pull vivek3369/customer-churn-prediction-app:latest

# 2. Run the container
docker run -p 8501:8501 vivek3369/customer-churn-prediction-app:latest
```
*The Streamlit App will instantly go live at `http://localhost:8501`*

---

### Option B: Run Locally (Developer Setup)

**Step 1: Clone the repository**
```bash 
git clone https://github.com/vivek-3369/end-to-end-customer-churn-prediction-project.git
cd end-to-end-customer-churn-prediction-project
```

**Step 2: Create a Conda environment**
```bash
conda create -p churn python==3.12 -y
conda activate churn/
```

**Step 3: Install the backend requirements**
```bash
pip install -r requirements.txt
```

*(Optional) If you face a ModuleNotFoundError regarding `src`, install the local package:*
```bash
pip install -e .
```

**Step 4: Run the Web Application**
```bash
streamlit run app.py
```
*The app will automatically open in your default web browser.*