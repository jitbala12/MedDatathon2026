import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

import streamlit as st

from src.data_loader import load_wdbc


WDBC_FEATURE_NAMES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


MODEL_PATHS = {
    "Logistic Regression (L1)": Path("models/best_logreg_l1.joblib"),
    "SVM (RBF kernel)": Path("models/best_svm_rbf.joblib"),
    "Random Forest": Path("models/best_rf.joblib"),
}


st.set_page_config(
    page_title="Breast Cancer Prediction (WDBC)",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    X, y = load_wdbc()

    # Attach canonical WDBC feature names for nicer display, preserving order
    if X.shape[1] == len(WDBC_FEATURE_NAMES):
        X = X.copy()
        X.columns = WDBC_FEATURE_NAMES

    df = X.copy()
    df["diagnosis"] = y.map({1: "Malignant", 0: "Benign"})
    return df, list(X.columns)


@st.cache_resource
def load_model(model_label: str):
    model_path = MODEL_PATHS[model_label]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            "Run the training script to generate it "
            "(e.g. `python -m src.train`)."
        )

    return joblib.load(model_path)


def build_user_input(df: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
    st.sidebar.header("Enter tumor measurements")

    user_values: dict[str, float] = {}
    for feature in feature_names:
        col_data = df[feature]
        min_val = float(col_data.min())
        max_val = float(col_data.max())
        default_val = float(col_data.median())

        user_values[feature] = st.sidebar.slider(
            feature,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
        )

    return user_values


def make_prediction(model, user_values: dict[str, float], feature_names: list[str]):
    values = [user_values[f] for f in feature_names]
    X_input = np.array(values, dtype=float).reshape(1, -1)

    proba_malignant = float(model.predict_proba(X_input)[0, 1])
    pred_label = int(proba_malignant >= 0.5)

    return proba_malignant, pred_label


def show_prediction_section(
    proba_malignant: float,
    pred_label: int,
) -> None:
    st.subheader("Prediction")

    st.metric(
        "Estimated probability of malignancy",
        f"{proba_malignant:.1%}",
    )

    if pred_label == 1:
        st.error(
            "The model prediction is **Malignant (cancerous)**.\n\n"
            "This is an algorithmic output and **must not** be used as a "
            "medical diagnosis. Please consult a clinician."
        )
    else:
        st.success(
            "The model prediction is **Benign (non-cancerous)**.\n\n"
            "This is an algorithmic output and **must not** be used as a "
            "medical diagnosis. Please consult a clinician."
        )


def show_dataset_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Diagnosis distribution**")
        counts = df["diagnosis"].value_counts().rename("count")
        st.bar_chart(counts)

    with col2:
        st.markdown("**Sample of the data**")
        st.dataframe(df.head(), use_container_width=True)


def show_feature_distribution(
    df: pd.DataFrame,
    user_values: dict[str, float],
    feature_names: list[str],
) -> None:
    st.subheader("Feature distribution")

    feature = st.selectbox(
        "Choose a feature to explore",
        feature_names,
    )

    fig, ax = plt.subplots()

    for label, color in [("Benign", "tab:blue"), ("Malignant", "tab:red")]:
        subset = df[df["diagnosis"] == label][feature]
        ax.hist(
            subset,
            bins=20,
            alpha=0.5,
            label=label,
            color=color,
        )

    if feature in user_values:
        ax.axvline(
            user_values[feature],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Your value",
        )

    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.legend()

    st.pyplot(fig)


def main() -> None:
    st.title("Breast Cancer Prediction (WDBC)")
    st.write(
        "Interactive app built on the Wisconsin Diagnostic Breast Cancer (WDBC) "
        "dataset. Provide tumor measurements to estimate the probability that "
        "a tumor is malignant vs benign.\n\n"
        "**This tool is for educational purposes only and is not a "
        "medical device.**"
    )

    df, feature_names = load_dataset()

    model_label = st.sidebar.selectbox(
        "Choose trained model",
        list(MODEL_PATHS.keys()),
    )

    try:
        model = load_model(model_label)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    user_values = build_user_input(df, feature_names)

    if st.sidebar.button("Predict diagnosis"):
        proba_malignant, pred_label = make_prediction(
            model,
            user_values,
            feature_names,
        )
        show_prediction_section(proba_malignant, pred_label)

    # Visualizations
    show_dataset_overview(df)
    show_feature_distribution(df, user_values, feature_names)


if __name__ == "__main__":
    main()

