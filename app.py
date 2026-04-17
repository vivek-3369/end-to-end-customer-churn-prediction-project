import streamlit as st
import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import json
import pickle
import warnings as wn
wn.filterwarnings("ignore")

from src.pipelines.prediction_pipeline import CustomData, PredictionPipeline


st.set_page_config(page_title="Churn Prediction", page_icon="📈")
st.title("📊 Customer Churn Prediction App")


@st.cache_data
def load_data():
    '''
    This function is responsible to load the csv file and return as dataframe
    '''
    df = pd.read_csv("artifacts/raw.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    return df

@st.cache_data
def load_metrics():
    '''
    This function is responsible to load and return metrics.json file
    '''
    with open("artifacts/metrics.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_model():
    with open("artifacts/model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_preprocessor():
    with open("artifacts/preprocessor.pkl", "rb") as f:
        return pickle.load(f)

def label_segment(score):
    """
    This function is used for labeling the RFM Score
    """
    if score >= 13 :
        return "Champions"
    elif score >= 10 :
        return "Loyal"
    elif score >=7 :
        return "At Risk"
    else :
        return "About to Churn"


df = load_data()
metrics = load_metrics()
model = load_model()
preprocessor = load_preprocessor()

# Creating Tabs for the page
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "🎯 Customer Segmentation (RFM)", "📈 Model Performance", "🔍 Key Insights From EDA"])

# 1. Prediction Tab
with tab1:
    st.markdown("Enter the customer profile details below to predict their churn likelihood 🔍")
    col1, col2 = st.columns(2)

    with col1 :
        gender = st.radio("Gender", ["Male", "Female"])
        seniorcitizen = st.radio("Senior Citizen", ["Yes", "No"])
        partner = st.radio("Partner", ["Yes", "No"])
        dependents = st.radio("Dependents", ["Yes", "No"])
        phoneservice = st.radio("Phone Service", ["Yes", "No"])
        multiplelines = st.radio("Multiple Lines", ["Yes", "No"])
        onlinesecurity = st.radio("Online Security", ["Yes", "No"])

    with col2:
        tenure = st.slider("Tenure", 1, 72, 1)
        deviceprotection = st.radio("Device Protection", ["Yes", "No"])
        techsupport = st.radio("Tech Support", ["Yes", "No"])
        streamingtv = st.radio("Streaming TV", ["Yes", "No"])
        streamingmovies = st.radio("Streaming Movies", ["Yes", "No"])
        paperlessbilling = st.radio("Paperless Billing", ["Yes", "No"])
        onlinebackup = st.radio("Online Backup", ["Yes", "No"])

    internetservice = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check","Bank transfer (automatic)" , "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges", min_value=10.00, max_value=500.00, value=10.00, format="%.2f")
    total_charges = st.number_input("Total Charges", min_value=10.00, max_value=100000.00, value=10.00, format="%.2f")


    if st.button("Predict Churn 📉") :
        user_data = CustomData(gender, seniorcitizen, partner, dependents, tenure, phoneservice, 
                        multiplelines, internetservice, onlinesecurity, onlinebackup, deviceprotection, 
                        techsupport, streamingtv, streamingmovies, contract, paperlessbilling, 
                        payment, monthly_charges, total_charges)
        
        # Creating a data frame from the user inputs
        data_df = user_data.get_input_as_df()

        # Sending the dataframe to prediction pipeline
        prediction_pipeline = PredictionPipeline()
        prob = prediction_pipeline.prediction(data_df)

        # To display the Churn Probabiltity
        st.divider()
        st.markdown("### 📋 Prediction Results")
        st.metric("Churn Probability:", f"{prob * 100:.2f}%")

        if prob > 0.6 :
            st.error("🚨 High Risk of Churn: Customer is likely to leave.")

        elif prob > 0.3 :
            st.warning("⚠️ Moderate Risk of Churn: Consider targeted retention.")
        else :
            st.success("✅ Low Risk of Churn: Customer is likely to stay.")

# 2. Customer Segmentation (RFM)
with tab2:
    # Creating services count column
    addon_services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    df["services_count"] = df[addon_services].apply(lambda row: (row=="Yes").sum(), axis=1)

    # Performing RFM(Recency, Frequency, Monetary Value) Analysis
    df["R_Score"] = pd.qcut(df["tenure"], q=5, labels=[1,2,3,4,5], duplicates="drop") # Low tenure = High Risk -> Score 5
    df["F_Score"] = pd.cut(df["services_count"], bins=[-1, 1, 2, 3, 4, 6], labels=[1,2,3,4,5]) # More Services = Score 5
    df["M_Score"] = pd.qcut(df["MonthlyCharges"], q=5, labels=[1,2,3,4,5], duplicates="drop")

    # Creating RFM Segment 
    df["RFM_Score"] = df["R_Score"].astype(int) + df["F_Score"].astype(int) + df["M_Score"].astype(int)
    df["Segment"] = df["RFM_Score"].apply(label_segment)


    st.markdown("""
        | Dimension | Source Column | Logic |
        |---|---|---|
        | **Recency (R)** | tenure | Lower tenure = newer = higher churn risk → scored 1-5) |
        | **Frequency (F)** | Count of active services | More services = more engaged → scored 1–5 |
        | **Monetary (M)** | MonthlyCharges | Higher spend = more valuable → scored 1–5 |
        """)
    st.markdown("### Customer Breakdown by Segment")

    fig, ax = plt.subplots(figsize=(5,4))
    labels = df["Segment"].value_counts().index
    values = (df["Segment"].value_counts(normalize=True)*100).values 
    plt.pie(values, labels=labels, autopct="%1.2f%%", 
        colors=["#C44A3A", "#D97A2B", "#F2D479", "#6FAF4F"],startangle=90, 
        wedgeprops={'edgecolor': 'black'})
    plt.legend(loc="upper left", fontsize=6)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


    st.markdown("### Average Churn Rate by Segments")

    df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)
    churn_grouped = (df.groupby("Segment")["Churn_binary"].mean() * 100).round(2)
    segment_labels = churn_grouped.index
    segment_churn_rate = churn_grouped.values

    fig, ax = plt.subplots(figsize=(5,4))
    sns.barplot(x=segment_labels, y=segment_churn_rate, ax=ax, palette=["#003049", "#d62828", "#f77f00", "#219ebc"], edgecolor="black")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Average Churn Rate (%)") 
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# 3. Model Performance     
with tab3 :
    st.subheader("Model Performance")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['Accuracy']*100:.1f}%")
    c2.metric("Precision", f"{metrics['Precision']*100:.1f}%")
    c3.metric("Recall", f"{metrics['Recall']*100:.1f}%")
    c4.metric("F1 Score", f"{metrics['F1_Score']*100:.1f}%")
    c5.metric("ROC-AUC Score", f"{metrics['ROC-AUC Score']*100:.1f}%")
 
    st.markdown("### Confusion Matrix")
    cm = np.array(metrics["Confusion Matrix"])
    labels= ["Stayed", "Churned"]

    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(
        cm, annot=True,fmt="d",
        xticklabels=labels, yticklabels=labels,cmap="crest", ax=ax, 
        edgecolor="black", linewidths=1
    ) 
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Metrics Chart")
    metrics_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC Score"]
    metrics_values = [metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["F1_Score"], metrics["ROC-AUC Score"]]


    fig,ax = plt.subplots(figsize=(5,4))
    sns.barplot(x=metrics_labels, y=metrics_values,palette="tab10", ax=ax, edgecolor="black")
    ax.set_title("Model Evaluation Metrics")
    ax.set_xticklabels(labels=metrics_labels, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close()
    
    st.markdown("### What do these metric means ?")
    st.markdown("""
        **Accuracy (76%):** Percentage of predictions that are correct.

        **Precision (54%):** Out of all predicted churners, how many are actually churned.

        **Recall (83%):** Out of all predicted churners, how many did we catch.

        **F1-Score (65%):** Harmonic Mean of Precision and Recall.

        **ROC-AUC Score (86%):** Model Ability to differentiate churners from non-churners.

        - **High Recall (83%)** means the model misses very few real churners — critical for a retention campaign.
        - The high ROC-AUC (86%) confirms the model ranks customers very well by churn risk.
        """)


# 4. Key Insights
with tab4 :
    st.title("Key Insights")
    st.markdown("""
    **Tenure is the Strongest Predictor:** Customers within their first 12 months (New Customers) have the highest churn rate. As tenure increases, churn probability drops dramatically.

    **Contract Type Matters:** Customers on a "Month-to-month" contract are vastly more likely to churn compared to those securely locked into "One-year" or "Two-year" contracts.

    **Payment Methods:** Customers utilizing "Electronic Check" as their primary payment method exhibited a disproportionately high churn rate compared to those using automatic billing. 

    **High Monthly Charges:** There is a strong positive correlation between high monthly charges and customer churn, particularly for those lacking bundled services like Tech Support or Online Security.

    **Senior Citizens & Vulnerability:** Senior citizens, particularly those without partners or dependents, are noticeably more vulnerable to churning.
    """)