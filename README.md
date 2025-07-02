Customer Churn Prediction
A machine learning project to predict customer churn for a telecom company, featuring a user-friendly Streamlit web application with interactive prediction and data analysis dashboards.

🚀 Overview
Customer churn is when a customer stops using a company’s service. Predicting churn helps businesses proactively retain customers and reduce revenue loss.
This project uses real-world telecom data, robust preprocessing, class balancing (SMOTE), and a Random Forest model to predict churn. The deployed Streamlit app allows both business users and data scientists to interactively assess churn risk and explore key patterns.

📂 Project Structure
text
├── app.py                 # Streamlit web app (prediction + analysis)
├── train_model.py         # Model training and preprocessing pipeline
├── churn_model.pkl        # Saved trained model
├── preprocessor.pkl       # Saved preprocessing objects (label encoders, etc.)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── requirements.txt       # Python dependencies
└── README.md              # This file
📝 Dataset
Source: WA_Fn-UseC_-Telco-Customer-Churn.csv

Rows: 7,043 customers

Columns: 21 features including demographics, contract info, service usage, and churn label.

🛠️ Features
Data Cleaning: Handles missing values, converts data types, drops irrelevant columns.

Feature Engineering: Label encoding for categorical variables.

Class Imbalance Handling: SMOTE oversampling for minority class (churners).

Modeling: Decision Tree, Random Forest, and XGBoost; Random Forest selected for best performance.

Evaluation: Precision, recall, F1-score, confusion matrix, and business-focused metrics.

Deployment: Streamlit app with:

Churn Prediction: Enter customer details, get risk assessment, probabilities, and a gauge visualization.

Data Analysis: Interactive filters, churn distribution, average charges by tenure group, and churn rate by contract.

🖥️ How to Run
Clone the repository and install dependencies:

bash
pip install -r requirements.txt
Train the model (if not already trained):

bash
python train_model.py
Run the Streamlit app:

bash
streamlit run app.py
Open your browser to http://localhost:8501

📊 App Demo
Predict Churn:
Input customer features, get instant churn prediction, probability, and a color-coded risk gauge.

Data Analysis:
Explore churn patterns with interactive filters and business-friendly charts (pie, bar, etc.).

🔑 Key Concepts
SMOTE: Synthetic oversampling to balance churn/no-churn classes.

Random Forest: Chosen for its accuracy and robustness on tabular data.

Label Encoding: Ensures categorical features are model-ready and consistent at inference.

Streamlit: Enables rapid web deployment for ML models and dashboards.

📈 Example Visualizations
Churn Distribution Pie Chart

Average Monthly Charges by Tenure Group (Bar Chart)

Churn Rate by Contract Type (Bar Chart)

Interactive risk gauge for prediction

🧑‍💻 Author
Ketki Dighe

📄 License
This project is for educational and demonstration purposes.

References:

CustomerChurnPrediction.ipynb

WA_Fn-UseC_-Telco-Customer-Churn.csv

Streamlit Documentation
