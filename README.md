Customer Churn Prediction

    This project predicts customer churn for a telecom company using machine learning and provides an interactive Streamlit web app for predictions and data analysis.

Overview
  
    Customer churn refers to customers leaving a service. Predicting churn helps businesses retain customers and reduce revenue loss.
    This project uses a real-world telecom dataset, applies preprocessing and class balancing, trains a Random Forest model, and deploys the solution as a web application.

Project Structure

    text
      app.py                   # Streamlit web app (prediction and analysis)
      train_model.py           # Model training and preprocessing
      churn_model.pkl          # Saved trained model
      preprocessor.pkl         # Saved preprocessing objects
      WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
      requirements.txt         # Python dependencies
      README.md                # Project documentation
      
    Dataset
      File: WA_Fn-UseC_-Telco-Customer-Churn.csv
  
    Rows: 7,043 customers
    
    Columns: 21 features (demographics, contract info, service usage, churn label)

Features

    Data cleaning and preprocessing (handle missing values, encode categorical variables)
    Class balancing using SMOTE
    Model training (Decision Tree, Random Forest, XGBoost)
    Model evaluation (accuracy, precision, recall, F1-score, confusion matrix)
    Deployment as a Streamlit web app
    Prediction page: Input customer details, get churn risk and probability
    Data analysis page: Interactive filters and visualizations

How to Run

  Install dependencies:

    text
    pip install -r requirements.txt
    Train the model (if not already trained):
    
    text
    python train_model.py
    Run the Streamlit app:
    
    text
    streamlit run app.py
    Open your browser to http://localhost:8501

App Functionality

    Predict Churn: Enter customer features, get churn prediction and probability.
    
    Data Analysis: Filter and visualize churn distribution, average charges by tenure, churn rate by contract, and more.

Key Concepts

    SMOTE: Oversampling technique to balance the dataset.
    
    Random Forest: Chosen for best performance and robustness.
    
    Label Encoding: Converts categorical features to numeric.
    
    Streamlit: Framework for building interactive Python web apps.

Author

    Ketki Dighe
