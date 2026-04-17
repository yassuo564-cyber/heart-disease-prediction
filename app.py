import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# -------------------------------------------------
# Load and preprocess dataset
# -------------------------------------------------
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
def train_models():
    df = load_data()

    X = df.drop("num", axis=1)
    y = df["num"]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ANN model
    ann_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    ann_model.fit(X_train_scaled, y_train)
    y_pred_ann = ann_model.predict(X_test_scaled)

    # KNN model
    knn_model = KNeighborsClassifier(n_neighbors=7)
    knn_model.fit(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)

    # Metrics
    ann_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_ann),
        "Precision": precision_score(y_test, y_pred_ann, zero_division=0),
        "Recall": recall_score(y_test, y_pred_ann, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred_ann, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred_ann)
    }

    knn_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_knn),
        "Precision": precision_score(y_test, y_pred_knn, zero_division=0),
        "Recall": recall_score(y_test, y_pred_knn, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred_knn, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred_knn)
    }

    return ann_model, knn_model, scaler, feature_names, ann_metrics, knn_metrics


ann_model, knn_model, scaler, feature_names, ann_metrics, knn_metrics = train_models()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("📌 Project Summary")
st.sidebar.write("**Project Title:** Heart Disease Prediction Using Supervised Machine Learning")
st.sidebar.write("**Dataset:** UCI Heart Disease Dataset")
st.sidebar.write("**Models Compared:** ANN and KNN")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Final Model Performance")
st.sidebar.write(f"**ANN Accuracy:** {ann_metrics['Accuracy']:.2%}")
st.sidebar.write(f"**KNN Accuracy:** {knn_metrics['Accuracy']:.2%}")
st.sidebar.write(f"**ANN Precision:** {ann_metrics['Precision']:.2%}")
st.sidebar.write(f"**KNN Precision:** {knn_metrics['Precision']:.2%}")
st.sidebar.write(f"**ANN Recall:** {ann_metrics['Recall']:.2%}")
st.sidebar.write(f"**KNN Recall:** {knn_metrics['Recall']:.2%}")
st.sidebar.write(f"**ANN F1-Score:** {ann_metrics['F1-Score']:.2%}")
st.sidebar.write(f"**KNN F1-Score:** {knn_metrics['F1-Score']:.2%}")

st.sidebar.markdown("---")
if knn_metrics["Accuracy"] >= ann_metrics["Accuracy"]:
    st.sidebar.success("✅ Best Model: KNN")
else:
    st.sidebar.success("✅ Best Model: ANN")

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

comparison_df = pd.DataFrame({
    "Model": ["ANN", "KNN"],
    "Accuracy": [ann_metrics["Accuracy"], knn_metrics["Accuracy"]],
    "Precision": [ann_metrics["Precision"], knn_metrics["Precision"]],
    "Recall": [ann_metrics["Recall"], knn_metrics["Recall"]],
    "F1-Score": [ann_metrics["F1-Score"], knn_metrics["F1-Score"]]
})

st.write("### Comparison Table")
st.dataframe(
    comparison_df.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}"
    }),
    use_container_width=True
)

st.write("### Performance Comparison Chart")

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
ann_scores = [
    ann_metrics["Accuracy"],
    ann_metrics["Precision"],
    ann_metrics["Recall"],
    ann_metrics["F1-Score"]
]
knn_scores = [
    knn_metrics["Accuracy"],
    knn_metrics["Precision"],
    knn_metrics["Recall"],
    knn_metrics["F1-Score"]
]

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(metrics))
width = 0.35

ax.bar([i - width/2 for i in x], ann_scores, width=width, label="ANN")
ax.bar([i + width/2 for i in x], knn_scores, width=width, label="KNN")

ax.set_xticks(list(x))
ax.set_xticklabels(metrics)
ax.set_ylim(0.80, 0.95)
ax.set_ylabel("Score")
ax.set_title("Performance Comparison of ANN and KNN")
ax.legend()

st.pyplot(fig)

if knn_metrics["Accuracy"] >= ann_metrics["Accuracy"]:
    st.success(
        "KNN performed slightly better than ANN in terms of accuracy, precision, and F1-score, "
        "while both models achieved the same recall. Therefore, KNN was selected as the final model."
    )
else:
    st.success(
        "ANN performed slightly better than KNN and was selected as the final model."
    )

st.markdown("---")

# -------------------------------------------------
# Confusion Matrix Section
# -------------------------------------------------
st.subheader("📉 Confusion Matrices")

col1, col2 = st.columns(2)

with col1:
    st.write("### ANN Confusion Matrix")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        ann_metrics["Confusion Matrix"],
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax1
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

with col2:
    st.write("### KNN Confusion Matrix")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        knn_metrics["Confusion Matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax2
    )
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

st.markdown("---")

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.subheader("🩺 Enter Patient Information")

left_col, right_col = st.columns(2)

with left_col:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    dataset = st.selectbox("Dataset Source", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=700, value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [False, True])

with right_col:
    restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
    thalch = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise-Induced Angina", [False, True])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thal", ["normal", "fixed defect", "reversable defect"])

selected_model = st.selectbox("Choose Prediction Model", ["ANN", "KNN"])

# Create input dataframe
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

# Apply same encoding as training
input_encoded = pd.get_dummies(input_data, drop_first=True)
input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_encoded)

# -------------------------------------------------
# Predict Button
# -------------------------------------------------
if st.button("🔍 Predict", use_container_width=True):
    if selected_model == "ANN":
        prediction = ann_model.predict(input_scaled)[0]
        model_used = "Artificial Neural Network (ANN)"
    else:
        prediction = knn_model.predict(input_scaled)[0]
        model_used = "K-Nearest Neighbors (KNN)"

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Prediction Result: Heart Disease Detected using **{model_used}**")
    else:
        st.success(f"✅ Prediction Result: No Heart Disease Detected using **{model_used}**")

    st.markdown("---")
    st.subheader("📋 Patient Input Summary")
    st.dataframe(input_data, use_container_width=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("BMCS2203 Artificial Intelligence | Heart Disease Prediction using ANN and KNN")
