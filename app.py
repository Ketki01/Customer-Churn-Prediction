import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go  # Required for gauge chart

# Load model and preprocessor
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load data for visualization
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})  # Convert to binary for plots

# --- NAVIGATION BAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Churn", "Data Analysis"])

# --- PREDICTION PAGE ---
if page == "Predict Churn":
    st.title("Customer Churn Prediction")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col2:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=0)
    monthly_charges = st.slider("Monthly Charges", min_value=0.0, max_value=1000.0, value=0.0)
    total_charges = st.slider("Total Charges", min_value=0.0, max_value=10000.0, value=0.0)

    # Gather inputs into a DataFrame
    input_dict = {
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    }
    input_df = pd.DataFrame(input_dict)

    # Apply preprocessing
    label_encoders = preprocessor['label_encoders']
    for col in label_encoders:
        if col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col])

    if st.button("Predict Churn"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[0]  # [prob_no_churn, prob_churn]

        st.header(" ")

        if prediction[0] == 1:
                st.error("HIGH CHURN RISK")
                st.caption("Immediate action recommended")
        else:
                st.success("LOW CHURN RISK")
                st.caption("Customer likely to stay")    

        col1, col2 = st.columns(2)
        
        with col1:
            churn_prob = prediction_proba[1] * 100
            st.metric("Churn Probability", f"{churn_prob:.1f}%")
            st.caption("Enhanced accuracy with SMOTE")
        
        with col2:
            retention_prob = prediction_proba[0] * 100
            st.metric("Retention Probability", f"{retention_prob:.1f}%")
            st.caption("Balanced prediction confidence")
        
        # Enhanced probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = churn_prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        # Display the gauge chart
        st.plotly_chart(fig, use_container_width=True)

# --- DATA ANALYSIS PAGE ---
elif page == "Data Analysis":
    st.title("📊 Customer Churn Analysis")
    
    # Dynamic Filters
    st.sidebar.header("Filter Data")
    selected_contract = st.sidebar.multiselect("Contract Type", df['Contract'].unique(), df['Contract'].unique())
    selected_payment = st.sidebar.multiselect("Payment Method", df['PaymentMethod'].unique(), df['PaymentMethod'].unique())
    
    # Apply filters
    filtered_df = df[
        df['Contract'].isin(selected_contract) & 
        df['PaymentMethod'].isin(selected_payment)
    ]
    
    # Key Metrics
    num_customers = len(filtered_df)
    churn_rate = filtered_df['Churn'].mean() * 100
    avg_tenure = filtered_df['tenure'].mean()
    avg_charge = filtered_df['MonthlyCharges'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", num_customers)
    col2.metric("Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Avg Tenure", f"{avg_tenure:.1f} months")
    col4.metric("Avg Monthly Charge", f"${avg_charge:.2f}")
    
    # Interactive Plots
    st.subheader("Churn Distribution")
    fig1 = px.pie(filtered_df, names='Churn', hole=0.3, 
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig1, use_container_width=True)
    
    # --- New Bar Chart: Average Monthly Charges by Tenure Group ---
    st.subheader("Average Monthly Charges by Tenure Group")
    # Bin tenure into intervals
    filtered_df['tenure_group'] = pd.cut(filtered_df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                                         labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
    avg_charges_by_tenure = filtered_df.groupby('tenure_group')['MonthlyCharges'].mean().reset_index()
    fig2 = px.bar(
        avg_charges_by_tenure,
        x='tenure_group',
        y='MonthlyCharges',
        labels={'tenure_group': 'Tenure (months)', 'MonthlyCharges': 'Avg Monthly Charges'},
        color='tenure_group',
        text_auto='.2f'
    )
    fig2.update_traces(textposition='outside')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Churn Rate by Contract Type")
    contract_churn = filtered_df.groupby('Contract')['Churn'].mean().reset_index()
    fig3 = px.bar(contract_churn, x='Contract', y='Churn', color='Contract',
                 labels={'Churn': 'Churn Rate'}, text_auto='.1%')
    fig3.update_traces(textposition='outside')
    st.plotly_chart(fig3, use_container_width=True)