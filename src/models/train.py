from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report
)


def train_models(X_train, y_train, preprocessor):

    models = {

        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ))
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    random_state=42
                ))
            ]
        )

    }

    trained_models = {}
    cv_results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"{name} trained successfully.")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return trained_models, cv_results


def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test set using multiple metrics.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of evaluation results for each model
    """
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred)
        }

        print(f"\n{name} - Test Set Evaluation:")
        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"Precision: {results[name]['precision']:.4f}")
        print(f"Recall: {results[name]['recall']:.4f}")
        print(f"ROC-AUC: {results[name]['roc_auc']:.4f}")
        print(f"\nClassification Report:\n{results[name]['classification_report']}")

    return results


def tune_hyperparameters(X_train, y_train, preprocessor):
    """
    Perform hyperparameter tuning for both models using GridSearchCV.

    Args:
        X_train: Training features
        y_train: Training labels
        preprocessor: Preprocessing pipeline

    Returns:
        Dictionary of best models after tuning
    """
    # Logistic Regression hyperparameters
    lr_params = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['liblinear', 'lbfgs']
    }

    # Random Forest hyperparameters
    rf_params = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # Create pipelines
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # Grid Search for Logistic Regression
    print("\nTuning Logistic Regression...")
    lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=5, scoring='roc_auc', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    print(f"Best Logistic Regression params: {lr_grid.best_params_}")
    print(f"Best ROC-AUC: {lr_grid.best_score_:.4f}")

    # Grid Search for Random Forest
    print("\nTuning Random Forest...")
    rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print(f"Best Random Forest params: {rf_grid.best_params_}")
    print(f"Best ROC-AUC: {rf_grid.best_score_:.4f}")

    return {
        "Logistic Regression (Tuned)": lr_grid.best_estimator_,
        "Random Forest (Tuned)": rf_grid.best_estimator_
    }
