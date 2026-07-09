import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from sklearn.model_selection import cross_val_score


def evaluate_models(models, X_train, X_test, y_train, y_test):

    results = []

    for name, model in models.items():

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        cv_score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        ).mean()

        results.append({

            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1 Score": f1_score(y_test, predictions),
            "ROC AUC": roc_auc_score(y_test, probabilities),
            "CV Accuracy": cv_score

        })

    return pd.DataFrame(results)


def save_feature_importance(model, feature_names):
    """
    Save Random Forest feature importance plot.

    Args:
        model: Trained Pipeline containing a RandomForestClassifier.
        feature_names (list): Original feature names.
    """

    classifier = model.named_steps["classifier"]

    # Check if classifier has feature_importances_ attribute
    if not hasattr(classifier, "feature_importances_"):
        print("Classifier does not have feature_importances_. "
              "Skipping feature importance plot.")
        return

    # Get feature importances
    importances = classifier.feature_importances_

    # Check if we need to handle OneHotEncoder transformed features
    if len(importances) != len(feature_names):
        # Create generic feature names for transformed features
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance = pd.Series(
        importances,
        index=feature_names
    ).sort_values(ascending=True)

    os.makedirs(
        "artifacts/reports",
        exist_ok=True
    )

    plt.figure(figsize=(8, 6))

    importance.plot(kind="barh")

    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")

    plt.tight_layout()

    plt.savefig(
        "artifacts/reports/feature_importance.png",
        dpi=300
    )

    plt.close()

    print("Feature importance plot saved successfully.")


def save_classification_report(model, X_test, y_test):
    """
    Generate and save the classification report.

    Args:
        model: Trained Pipeline.
        X_test: Test features.
        y_test: Test labels.
    """

    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        digits=4
    )

    os.makedirs(
        "artifacts/reports",
        exist_ok=True
    )

    report_path = "artifacts/reports/classification_report.txt"

    with open(report_path, "w") as file:
        file.write("Classification Report\n")
        file.write("=" * 60)
        file.write("\n\n")
        file.write(report)

    print(f"Classification report saved to {report_path}")
