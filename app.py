import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Customer Churn Predictor")
# Load the pre-trained model
model = joblib.load('cust_churn_model.pkl')

st.title("Customer Churn Predictor")

st.write("Input the feature values below to get a prediction from the model.")


# Create input elements for features
feature_1 = st.selectbox("Gender:", ['', 'Male', 'Female'])
feature_2 = st.selectbox("SeniorCitizen:", ['', 'Y', 'N'])
feature_3 = st.selectbox("Partner:", ['', 'Yes', 'No'])
feature_4 = st.selectbox("Dependents:", ['', 'Yes', 'No'])
feature_5 = st.number_input("tenure:", value=0)
feature_6 = st.selectbox("PhoneService:", ['', 'Yes', 'No'])
feature_7 = st.selectbox("MultipleLines:", ['', 'Yes', 'No', 'No phone service'])
feature_8 = st.selectbox("InternetService:", ['', 'Fiber optic', 'DSL', 'No'])
feature_9 = st.selectbox("OnlineSecurity:", ['', 'Yes', 'No', 'No internet service'])
feature_10 = st.selectbox("OnlineBackup:", ['', 'Yes', 'No', 'No internet service'])
feature_11 = st.selectbox("DeviceProtection:", ['', 'Yes', 'No', 'No internet service'])
feature_12 = st.selectbox("TechSupport:", ['', 'Yes', 'No', 'No internet service'])
feature_13 = st.selectbox("StreamingTV:", ['', 'Yes', 'No', 'No internet service'])
feature_14 = st.selectbox("StreamingMovies:", ['', 'Yes', 'No', 'No internet service'])
feature_15 = st.selectbox("Contract:", ['', 'One year', 'Two year', 'Month-to-month'])
feature_16 = st.selectbox("PaperlessBilling:", ['', 'Yes', 'No'])
feature_17 = st.selectbox("PaymentMethod:", ['', 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
feature_18 = st.number_input("MonthlyCharges:", value=0.0)
feature_19 = st.number_input("TotalCharges:", value=0.0)


# Predict button
if st.button("Predict"):

    # Format inputs into the DataFrame structure
    input_data = pd.DataFrame({
        'gender': [feature_1],
        'SeniorCitizen': [feature_2],
        'Partner': [feature_3],
        'Dependents': [feature_4],
        'tenure': [feature_5],
        'PhoneService': [feature_6],
        'MultipleLines': [feature_7],
        'InternetService': [feature_8],
        'OnlineSecurity': [feature_9],
        'OnlineBackup': [feature_10],
        'DeviceProtection': [feature_11],
        'TechSupport': [feature_12],
        'StreamingTV': [feature_13],
        'StreamingMovies': [feature_14],
        'Contract': [feature_15],
        'PaperlessBilling': [feature_16],
        'PaymentMethod': [feature_17],
        'MonthlyCharges': [feature_18],
        'TotalCharges': [feature_19]
    })

    # Generate prediction
    prediction = model.predict(input_data)

    # Display the result to the user
    st.success(f"The model predicted: {prediction[0]}")