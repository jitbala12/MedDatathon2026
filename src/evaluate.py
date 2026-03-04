from typing import Dict
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    Returns dictionary of performance metrics.
    """

    # Predicted probabilities (for AUC)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Predicted classes
    y_pred = model.predict(X_test)

    metrics = {
        "auc": roc_auc_score(y_test, y_probs),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return metrics


def print_confusion(model, X_test, y_test):
    """
    Print confusion matrix separately.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)