def select_best_model(results, models):

    best_model_name = results.sort_values(
        by="ROC AUC",
        ascending=False
    ).iloc[0]["Model"]

    print(f"Best Model: {best_model_name}")

    return models[best_model_name]
