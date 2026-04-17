import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS for better appearance
# -------------------------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #c62828;
        text-align: center;
    }
    .sub-title {
        font-size: 18px;
        text-align: center;
        color: #555555;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #1f4e79;
        margin-top: 20px;
    }
    .box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown('<p class="main-title">❤️ Heart Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">This web application compares <b>Artificial Neural Network (ANN)</b> and '
    '<b>K-Nearest Neighbors (KNN)</b> models and predicts the presence of heart disease using the '
    '<b>best-performing model (KNN)</b>.</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------
# Model Comparison Section
# -------------------------------------------------
st.markdown('<p class="section-title">📊 Model Comparison</p>', unsafe_allow_html=True)

comparison_df = pd.DataFrame({
    "Model": ["ANN", "KNN"],
    "Accuracy": [0.8587, 0.8696],
    "Precision": [0.8455, 0.8611],
    "Recall": [0.9118, 0.9118],
    "F1-Score": [0.8774, 0.8857]
})

col1, col2 = st.columns([2, 1])

with col1:
    st.table(comparison_df)

with col2:
    st.info(
        "✅ **KNN performed slightly better than ANN**\n\n"
        "- Higher Accuracy\n"
        "- Higher Precision\n"
        "- Same Recall\n"
        "- Higher F1-Score\n\n"
        "**Selected Model for Deployment: KNN**"
    )

metric1, metric2, metric3, metric4, metric5 = st.columns(5)
metric1.metric("ANN Accuracy", "85.87%")
metric2.metric("KNN Accuracy", "86.96%")
metric3.metric("ANN F1-Score", "87.74%")
metric4.metric("KNN F1-Score", "88.57%")
metric5.metric("Best Model", "KNN")

st.markdown("""
The table above shows the performance of ANN and KNN on the UCI Heart Disease dataset.  
Although both models performed well, **KNN achieved slightly better overall performance** in terms of **accuracy, precision, and F1-score**, while both models obtained the same recall. Therefore, **KNN was selected as the final model** for deployment in this application.
""")

st.markdown("---")

# -------------------------------------------------
# Load and preprocess dataset
# -------------------------------------------------
df = pd.read_csv("heart_disease_uci.csv")

# Drop unnecessary column
df = df.drop("id", axis=1)

# Convert target to binary classification
df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

# Handle missing values
numerical_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal"]
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Feature and target
X = df.drop("num", axis=1)
y = df["num"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Final KNN model
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train, y_train)

# -------------------------------------------------
# User Input Section
# -------------------------------------------------
st.markdown('<p class="section-title">🩺 Enter Patient Information</p>', unsafe_allow_html=True)

st.markdown("""
Please enter the patient's clinical information below.  
The system will predict whether the patient is likely to have **heart disease** based on the **KNN model (k = 7)**.
""")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    dataset = st.selectbox("Dataset Source", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [False, True])

with col2:
    restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
    thalch = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise-Induced Angina", [False, True])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

# Create input DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "dataset": [dataset],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalch": [thalch],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

# Encode input
input_encoded = pd.get_dummies(input_data, drop_first=True)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_encoded)

st.markdown("")

# -------------------------------------------------
# Prediction Button
# -------------------------------------------------
if st.button("🔍 Predict"):
    prediction = knn_model.predict(input_scaled)[0]

    st.markdown('<p class="section-title">📌 Prediction Result</p>', unsafe_allow_html=True)

    if prediction == 1:
        st.error("⚠️ The patient is predicted to have **Heart Disease**.")
    else:
        st.success("✅ The patient is predicted to have **No Heart Disease**.")

    st.markdown("""
    **Note:** This prediction is generated using the **K-Nearest Neighbors (KNN)** model with **k = 7**,  
    which was selected as the **best-performing model** in this project.
    """)
