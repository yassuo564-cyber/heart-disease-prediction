import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("📌 Project Summary")
st.sidebar.write("**Project Title:** Heart Disease Prediction Using Supervised Machine Learning")
st.sidebar.write("**Dataset:** UCI Heart Disease Dataset")
st.sidebar.write("**Models Compared:** ANN and KNN")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance")
st.sidebar.write("**ANN Accuracy:** 85.87%")
st.sidebar.write("**KNN Accuracy:** 86.96%")
st.sidebar.write("**ANN Precision:** 84.55%")
st.sidebar.write("**KNN Precision:** 86.11%")
st.sidebar.write("**ANN Recall:** 91.18%")
st.sidebar.write("**KNN Recall:** 91.18%")
st.sidebar.write("**ANN F1-Score:** 87.74%")
st.sidebar.write("**KNN F1-Score:** 88.57%")

st.sidebar.markdown("---")
st.sidebar.success("✅ Best Model: KNN")

# -------------------------------------------------
# Main Title
# -------------------------------------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    This web application compares the performance of **Artificial Neural Network (ANN)**  
    and **K-Nearest Neighbors (KNN)** models and predicts the presence of  
    **heart disease** using either selected model.
    """
)

st.markdown("---")

# -------------------------------------------------
# Comparison Section
# -------------------------------------------------
st.subheader("📊 Model Comparison")

# Comparison Table
comparison_df = pd.DataFrame({
    "Model": ["ANN", "KNN"],
    "Accuracy": [0.8587, 0.8696],
    "Precision": [0.8455, 0.8611],
    "Recall": [0.9118, 0.9118],
    "F1-Score": [0.8774, 0.8857]
})

st.write("### Comparison Table")
st.table(comparison_df)

# Comparison Bar Chart
st.write("### Comparison Chart")

chart_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "ANN": [0.8587, 0.8455, 0.9118, 0.8774],
    "KNN": [0.8696, 0.8611, 0.9118, 0.8857]
})

fig, ax = plt.subplots(figsize=(8, 5))

x = range(len(chart_df["Metric"]))
width = 0.35

ax.bar([i - width/2 for i in x], chart_df["ANN"], width=width, label="ANN")
ax.bar([i + width/2 for i in x], chart_df["KNN"], width=width, label="KNN")

ax.set_xticks(list(x))
ax.set_xticklabels(chart_df["Metric"])
ax.set_ylim(0.80, 0.95)
ax.set_ylabel("Score")
ax.set_title("Performance Comparison of ANN and KNN")
ax.legend()

st.pyplot(fig)

# Best model explanation
st.info(
    """
    **Interpretation:**  
    The chart above shows that **KNN** achieved slightly better performance than **ANN**
    in terms of **accuracy, precision, and F1-score**, while both models obtained the same
    **recall**. Therefore, **KNN** was selected as the **best model** for this project.
    """
)

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

# Train ANN model
ann_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
ann_model.fit(X_train, y_train)

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train, y_train)

# -------------------------------------------------
# User Input Section
# -------------------------------------------------
st.subheader("🩺 Enter Patient Information")

left_col, right_col = st.columns(2)

with left_col:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    dataset = st.selectbox("Dataset Source", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [False, True])

with right_col:
    restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
    thalch = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise-Induced Angina", [False, True])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

# Model selection
selected_model = st.selectbox("Choose Prediction Model", ["ANN", "KNN"])

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

# One-hot encode input
input_encoded = pd.get_dummies(input_data, drop_first=True)

# Align with training data columns
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_encoded)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔍 Predict Heart Disease"):
    if selected_model == "ANN":
        prediction = ann_model.predict(input_scaled)[0]
        model_used = "Artificial Neural Network (ANN)"
    else:
        prediction = knn_model.predict(input_scaled)[0]
        model_used = "K-Nearest Neighbors (KNN)"

    st.markdown("---")
    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ The patient is predicted to have **Heart Disease** using **{model_used}**.")
    else:
        st.success(f"✅ The patient is predicted to have **No Heart Disease** using **{model_used}**.")

    st.write(f"This prediction is generated using the **{model_used}** model.")

    st.subheader("📋 Patient Input Summary")
    st.dataframe(input_data, use_container_width=True)
