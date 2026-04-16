import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load and preprocess dataset
# -----------------------------
@st.cache_data
def load_data():
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

    return df

@st.cache_resource
def train_model():
    df = load_data()

    X = df.drop("num", axis=1)
    y = df["num"]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Save feature names for alignment later
    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Best KNN model
    knn_model = KNeighborsClassifier(n_neighbors=7)
    knn_model.fit(X_train_scaled, y_train)

    return knn_model, scaler, feature_names

model, scaler, feature_names = train_model()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 About")
st.sidebar.info(
    """
    This Streamlit application predicts whether a patient is likely to have heart disease
    using the **K-Nearest Neighbours (KNN)** model.

    **Dataset:** UCI Heart Disease Dataset  
    **Selected model:** KNN (k = 7)  
    **Accuracy:** 86.96%  
    **Precision:** 86.11%  
    **Recall:** 91.18%  
    **F1-score:** 88.57%
    """
)

st.sidebar.markdown("---")
st.sidebar.write("Developed for BMCS2203 Artificial Intelligence Assignment")

# -----------------------------
# Main title
# -----------------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    This application predicts whether a patient is likely to have **heart disease**
    based on clinical input values using the **KNN classification model**.
    Please enter the patient's details below and click **Predict**.
    """
)

st.markdown("---")

# -----------------------------
# Input sections
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Demographic Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    dataset = st.selectbox(
        "Dataset Source",
        ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
    )
    cp = st.selectbox(
        "Chest Pain Type",
        ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
    )
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50, max_value=250, value=120
    )
    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=50, max_value=700, value=200
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [True, False]
    )

with col2:
    st.subheader("🩺 Clinical Information")
    restecg = st.selectbox(
        "Resting ECG",
        ["normal", "lv hypertrophy", "st-t abnormality"]
    )
    thalch = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=50, max_value=250, value=150
    )
    exang = st.selectbox(
        "Exercise Induced Angina",
        [True, False]
    )
    oldpeak = st.number_input(
        "Oldpeak (ST depression induced by exercise)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.1
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["upsloping", "flat", "downsloping"]
    )
    ca = st.number_input(
        "Number of Major Vessels (ca)",
        min_value=0.0, max_value=3.0, value=0.0, step=1.0
    )
    thal = st.selectbox(
        "Thal",
        ["normal", "fixed defect", "reversable defect"]
    )

st.markdown("---")

# -----------------------------
# Predict button
# -----------------------------
if st.button("🔍 Predict", use_container_width=True):
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "dataset": dataset,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Apply same encoding as training
    input_encoded = pd.get_dummies(input_data, drop_first=True)

    # Align columns with training features
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

    # Scale input
    input_scaled = scaler.transform(input_encoded)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Prediction Result: Heart Disease Detected")
        st.write(f"**Probability of Heart Disease:** {probability[1]*100:.2f}%")
    else:
        st.success("✅ Prediction Result: No Heart Disease Detected")
        st.write(f"**Probability of No Heart Disease:** {probability[0]*100:.2f}%")

    st.markdown("---")
    st.subheader("📋 Patient Input Summary")
    st.dataframe(input_data, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("BMCS2203 Artificial Intelligence | Heart Disease Prediction using KNN")
