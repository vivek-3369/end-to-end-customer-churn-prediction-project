# 📊 Telecom Customer Churn Prediction: End-to-End Machine Learning Pipeline

## 📝 Project Overview
Customer Churn is a critical metric for businesses, particularly in the telecommunications industry where acquiring a new customer is significantly more expensive than retaining an existing one. This project is a complete, end-to-end Machine Learning pipeline designed to predict whether a customer will discontinue their telecom services (churn) or remain loyal. By predicting churn probability, telecom companies can proactively identify at-risk customers and offer targeted retention strategies.

This project utilizes the **Telco Customer Churn dataset** from Kaggle and features a robust backend pipeline coupled with an interactive web application built with Streamlit.

## Dataset Description
The model is trained on the publicly available **Telco Customer Churn Database** from Kaggle. The dataset captures detailed metrics of over 7000 customers. Key features include:

- **Demographics:** Gender, Senior Citizen status, Partners, and Dependents.
- **Account Information:** Tenure (months with the company), Contract type, Payment method, Paperless billing.
- **Services Subscribed:** Phone service, Multiple lines, Internet service, Online security, Tech support, Streaming TV/Movies.
- **Financial Details:** Monthly charges and Total charges.
- **Target Variable:** Churn (Yes/No).

---

## 🛠️ Technology Stack
* **Programming Language:** Python
* **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE)
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Deployment & Web App:** Streamlit
* **Environment Management:** Conda

---

## ⚙️ Architecture & Pipeline Design
The project is built on a highly modular and scalable architecture divided into two primary pipelines:

### 1. Training Pipeline
The training pipeline automates the entire process of learning from historical data. It is constructed using the following modular components:
* **Data Ingestion:** Reads the raw data from local sources, performs a train-test split, and outputs the data structures into the `artifacts/` folder to ensure data consistency.
* **Data Transformation:** Handles data cleaning and extensive Feature Engineering. Missing values are imputed, categorical variables are encoded (One-Hot & Ordinal), and numerical features are scaled. To handle severe class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** is correctly applied exclusively to the training set to prevent data leakage. The entire transformation process is saved as a `preprocessor.pkl` object.
* **Model Training:** Evaluates multiple baseline algorithms including Logistic Regression, Random Forest, Decision Trees, K-Nearest Neighbors, AdaBoost, and Gradient Boosting. The best-performing model is saved as `model.pkl`.
* **Model Evaluation:** Evaluates the best model on unseen testing data. It calculates crucial classification metrics such as Accuracy, Precision, Recall, and ROC-AUC score, and ultimately stores these metrics in a JSON file for CI/CD tracking.

### 2. Prediction Pipeline
The prediction pipeline acts as the bridge between the trained model and the user interface. It intercepts raw user input from the front end, dynamically applies the same feature engineering logic used during training, scales the inputs using the saved `preprocessor.pkl`, and passes the data to the `model.pkl` to calculate the real-time probability of customer churn.

---

## Web Application & Deployment
The Streamlit application (`app.py`) is styled to replicate a professional data dashboard. Using interactive form components such as `st.selectbox`, `st.number_input`, and `st.slider`, network administrators can manually construct a customer's profile.

Upon prediction, the application displays the calculated percentage likelihood of churn and initiates conditional warnings:
- **Low Risk:** Probability below 30% generates a green active success metric.
- **Moderate Risk:** Probability between 30% - 60% generates a yellow retention warning.
- **High Risk:** Probability exceeding 60% triggers a severe red critical alert recommending immediate organizational intervention.

---

## 🔍 Key Insights from Exploratory Data Analysis (EDA)
During the initial EDA phase, several critical patterns regarding customer behavior were discovered:

* **Tenure is the Strongest Predictor:** Customers within their first 12 months (New Customers) have the highest churn rate. As tenure increases, churn probability drops dramatically.
* **Contract Type Matters:** Customers on a "Month-to-month" contract are vastly more likely to churn compared to those securely locked into "One-year" or "Two-year" contracts.
* **Payment Methods:** Customers utilizing "Electronic Check" as their primary payment method exhibited a disproportionately high churn rate compared to those using automatic billing. 
* **High Monthly Charges:** There is a strong positive correlation between high monthly charges and customer churn, particularly for those lacking bundled services like Tech Support or Online Security.
* **Senior Citizens & Vulnerability:** Senior citizens, particularly those without partners or dependents, are noticeably more vulnerable to churning.

---

## 💼 Business Value & Recommendations
Integrating this predictive model into a telecom company's daily operations can deliver immense business value. 

### Benefits to the Business:
1. **Cost Reduction:** Retaining an existing customer is significantly cheaper than spending marketing capital to simply acquire a new one.
2. **Resource Optimization:** By identifying exact customers at risk, customer service teams can focus their outreach efforts efficiently rather than offering blanket discounts to everyone.
3. **Revenue Forecasting:** Understanding future churn allows for more accurate revenue projection and financial planning.

### Strategic Recommendations:
1. **Targeted First-Year Interventions:** Since month-to-month, low-tenure customers churn the most, the business should offer heavy discounts or free premium upgrades to customer migrating them to 1-year contracts at the 6-month mark.
2. **Promote Automatic Payments:** Establish cash-back incentives or loyalty points for customers who switch from Electronic Checks to automatic Credit Card or Bank Transfer payments to create billing friction for leaving.
3. **Bundle "Sticky" Services:** Customers with Tech Support and Online Security rarely leave. Marketing should push customized bundles offering these services for a reduced price to high-risk customers.

---

## 🚀 Future Improvements
While the current model functions very well, this project can be expanded in the following ways:
* **Advanced Deep Learning Models:** Implement neural networks or XGBoost/CatBoost algorithms to capture more complex non-linear relationships in customer behavior.
* **Automated Data Drift Detection:** Integrate EvidentlyAI to monitor the incoming data format via the web app. If the distribution of user inputs shifts drastically from the training data, trigger an alert to retrain the model.
* **Cloud Database Integration:** Connect the Streamlit application to a cloud database (like MongoDB or AWS RDS) to log the predictions and actual outcomes, creating a feedback loop for continuous model improvement.
* **FastAPI Backend:** Separate the Streamlit frontend from the predictive modeling completely by wrapping the Prediction Pipeline inside a dedicated REST API using FastAPI.

---gi

## 💻 How to Run the Project Locally

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

**Step 3: Install the requirements**
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