import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st

from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.data_loader import load_wdbc

# ---------------------------------------------------
# Feature names
# ---------------------------------------------------

WDBC_FEATURE_NAMES = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean",
    "fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se",
    "smoothness_se","compactness_se","concavity_se","concave_points_se",
    "symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
    "perimeter_worst","area_worst","smoothness_worst","compactness_worst",
    "concavity_worst","concave_points_worst","symmetry_worst",
    "fractal_dimension_worst"
]

MODEL_PATHS = {
    "Logistic Regression": Path("models/best_logreg_l1.joblib"),
    "SVM (RBF)": Path("models/best_svm_rbf.joblib"),
    "Random Forest": Path("models/best_rf.joblib"),
}

# ---------------------------------------------------
# Page config
# ---------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Diagnosis App",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------

@st.cache_resource
def load_dataset():
    X, y = load_wdbc()
    X.columns = WDBC_FEATURE_NAMES
    df = X.copy()
    df["diagnosis"] = y
    return df


@st.cache_resource
def load_model(model_label):
    return joblib.load(MODEL_PATHS[model_label])


# ---------------------------------------------------
# ABOUT SECTION
# ---------------------------------------------------

def show_about():

    st.title("Breast Cancer Diagnosis Prediction")

    st.markdown("""
### Project Overview
This application predicts whether a breast tumor is **benign or malignant**
using machine learning models trained on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

The goal is to demonstrate how machine learning models can assist with
medical classification problems.

### Dataset
The dataset contains **569 tumor samples** with **30 numerical features**
derived from digitized images of breast cell nuclei.

Each sample is labeled as:

- **Benign (0)**
- **Malignant (1)**

### Feature Types
Features describe properties of the tumor such as:

- radius
- texture
- perimeter
- area
- smoothness
- concavity
- symmetry

These are computed as:

- mean
- standard error
- worst (largest value)

### Models Evaluated
Three ML models were trained using cross-validation:

- Logistic Regression (L1 regularization)
- Support Vector Machine (RBF kernel)
- Random Forest

Model selection was performed using **ROC-AUC**.

**Disclaimer:**  
This tool is for educational purposes only and must not be used for medical diagnosis.
""")


# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

def get_user_inputs(df):

    st.sidebar.header("Tumor Measurements")

    inputs = {}

    for feature in WDBC_FEATURE_NAMES:

        col = df[feature]

        inputs[feature] = st.sidebar.slider(
            feature,
            float(col.min()),
            float(col.max()),
            float(col.median())
        )

    return inputs


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

def predict(model, inputs):

    X = np.array(list(inputs.values())).reshape(1, -1)

    prob = model.predict_proba(X)[0,1]
    pred = int(prob >= 0.5)

    return prob, pred


def show_prediction(prob, pred):

    st.subheader("Prediction")

    st.metric("Probability of Malignancy", f"{prob:.2%}")

    if pred == 1:
        st.error("Prediction: **Malignant**")
    else:
        st.success("Prediction: **Benign**")


# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------

def show_metrics():

    st.header("Model Performance")

    metrics_path = Path("models/model_metrics.csv")

    if metrics_path.exists():
        results = pd.read_csv(metrics_path)
        st.dataframe(results, use_container_width=True)
    else:
        st.warning("Model metrics not found. Run training first.")

    st.dataframe(results, use_container_width=True)


# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------

def show_confusion(model, df):

    X = df[WDBC_FEATURE_NAMES].values
    y = df["diagnosis"].values

    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)


# ---------------------------------------------------
# ROC CURVE
# ---------------------------------------------------

def show_roc(df):

    st.header("ROC Curve Comparison")

    X = df[WDBC_FEATURE_NAMES].values
    y = df["diagnosis"].values

    fig, ax = plt.subplots()

    for name, path in MODEL_PATHS.items():

        model = joblib.load(path)

        probs = model.predict_proba(X)[:,1]

        fpr, tpr, _ = roc_curve(y, probs)

        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    ax.plot([0,1],[0,1],"--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.legend()

    st.pyplot(fig)


# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------

def main():

    df = load_dataset()

    show_about()

    st.sidebar.header("Model Selection")

    model_name = st.sidebar.selectbox(
        "Choose model",
        list(MODEL_PATHS.keys())
    )

    model = load_model(model_name)

    user_inputs = get_user_inputs(df)

    if st.sidebar.button("Predict Diagnosis"):

        prob, pred = predict(model, user_inputs)

        show_prediction(prob, pred)

    show_metrics()

    show_confusion(model, df)

    show_roc(df)


if __name__ == "__main__":
    main()