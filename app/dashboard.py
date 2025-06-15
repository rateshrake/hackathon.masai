import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# 🔧 1. Preprocess Function (from your notebook)
def preprocess(sample):
    sample['MultipleLines'] = sample['MultipleLines'].replace('No phone service', 'No')
    obj_cols = sample.select_dtypes(include='object').columns

    le_dict = {}

    for col in obj_cols:
        if col != "customerID":
            sample[col] = sample[col].astype(str)
            le = LabelEncoder()
            sample[col] = le.fit_transform(sample[col])
            le_dict[col] = le

    return sample

# 🔮 Load model
model = joblib.load("models/model.pkl")
st.title("Churn Probabilty Model")
st.subheader("🧪 Model Evaluation on Training Data")
try:
    train_df = pd.read_csv("data/data.csv")  
    y_train_true = train_df["churned"] 
    train_processed = preprocess(train_df.copy())
    X_train = train_processed.drop(columns=["customerID", "churned"], errors='ignore')  
    y_train_pred = model.predict_proba(X_train)[:, 1]
    # AUC Score
    from sklearn.metrics import roc_auc_score, roc_curve
    auc_train = roc_auc_score(y_train_true, y_train_pred)


    st.markdown(
    f"<h5 style='color:lightgreen;'>AUC-ROC Score on Training Data: <b>{auc_train:.4f}</b></h5>",
    unsafe_allow_html=True
    )
    fpr, tpr, _ = roc_curve(y_train_true, y_train_pred)
    fig_train_roc, ax = plt.subplots(facecolor='#001f3f')
    ax.set_facecolor('#001f3f')
    ax.plot(fpr, tpr, label=f"AUC = {auc_train:.2f}", color="white")
    ax.plot([0, 1], [0, 1], 'k--', color='gray', label="Random Model")

    ax.set_title("ROC Curve (Training Data)", color='white')
    ax.set_xlabel("False Positive Rate", color='white')
    ax.set_ylabel("True Positive Rate", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.legend()

    st.pyplot(fig_train_roc)

except Exception as e:
    st.warning(f"⚠️ Could not load training data for evaluation. Error: {e}")

# 🎯 Streamlit UI
st.subheader("📉 Customer Churn Prediction Dashboard")

uploaded_file = st.file_uploader("📁 Upload test data CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='latin1')

    st.subheader("📋 Raw Uploaded Data")
    st.dataframe(df.head())

    if 'customerID' in df.columns:
        customer_ids = df['customerID']
    else:
        st.warning("⚠️ 'customerID' column not found. Prediction output may not be traceable.")
        customer_ids = pd.Series([f"CUST_{i}" for i in range(len(df))])

    df_processed = preprocess(df.copy())
    df_processed = df_processed.drop(columns=['customerID'], errors='ignore')

    # 🔮 Predict
    churn_probs = model.predict_proba(df_processed)[:, 1]
    prediction_df = pd.DataFrame({
        'customerID': customer_ids,
        'Churn_Probability': churn_probs,
        'Churned': (churn_probs >= 0.5).astype(int)
    })


        # --- Load Training Data for Model Evaluation ---
   


    st.subheader("🔮 Predictions")
    st.dataframe(prediction_df.head(10))
    


    # Distribution Plot
    st.subheader("📊 Churn Probability Distribution")

    fig, ax = plt.subplots(facecolor='#001f3f')     
    ax.set_facecolor('#001f3f')                    

   
    sns.histplot(prediction_df['Churn_Probability'], kde=True, bins=25, ax=ax, color="white")

    ax.set_xlabel("Churn Probability", color='white')
    ax.set_ylabel("Frequency", color='white')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    st.pyplot(fig)

    
    st.subheader("📈 Churn vs Retain Pie Chart")
    pie_data = prediction_df['Churned'].value_counts().rename({0: "Retained", 1: "Churned"})
    fig, ax = plt.subplots(facecolor='#001f3f')
    ax.set_facecolor('#001f3f')
    colors = ['gold', "#FF0D00"]  
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=colors, textprops={'color': 'white'})
    st.pyplot(fig)


    #Top 10 Risk Table
    st.subheader("🚨 Top 10 High-Risk Customers")
    top10 = prediction_df.sort_values("Churn_Probability", ascending=False).head(10)
    st.dataframe(top10)

    # Download Button
    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=prediction_df.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )


st.markdown(
    """
    <div style="text-align: center;">
        <a href="https://drive.google.com/file/d/1v0c36TtF8vpndTHJPf-t_ACUC8U1dDUW/view?usp=sharing" target="_blank">
            <button style="
                background-color: red;
                color: white;
                padding: 20px 40px;
                font-size: 24px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
            ">
                for video presentation
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
