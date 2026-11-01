from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.dummy import DummyClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "outputs" / "lapse_dataset.csv"

NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_minutes",
    "tenure_days",
    "age",
]

CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "favourite_category",
]


def main():
    dataset = pd.read_csv(DATA_PATH)

    X = dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = dataset["lapsed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("DATA SPLIT CHECK")
    print("=" * 40)
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    print("\nBASELINE MODEL")
    print("Accuracy:", round(accuracy_score(y_test, baseline_predictions), 3))
    print(
        "Lapsed-class F1:",
        round(f1_score(y_test, baseline_predictions, zero_division=0), 3),
    )

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    parameter_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100],
    }

    print("\nTraining Logistic Regression with 5-fold cross-validation...")

    search = GridSearchCV(
        logistic_pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=5,
        n_jobs=1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    print("\nLOGISTIC REGRESSION RESULTS")
    print("=" * 40)
    print("Best parameters:", search.best_params_)
    print("Best cross-validation F1:", round(search.best_score_, 3))
    print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Test lapsed-class F1:", round(f1_score(y_test, predictions), 3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
 # Random Forest pipeline
    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )

    rf_parameter_grid = {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    }

    print("\nTraining Random Forest with 5-fold cross-validation...")

    rf_search = GridSearchCV(
        random_forest_pipeline,
        param_grid=rf_parameter_grid,
        scoring="f1",
        cv=5,
        n_jobs=1,
    )

    rf_search.fit(X_train, y_train)

    rf_model = rf_search.best_estimator_
    rf_predictions = rf_model.predict(X_test)

    print("\nRANDOM FOREST RESULTS")
    print("=" * 40)
    print("Best parameters:", rf_search.best_params_)
    print("Best cross-validation F1:", round(rf_search.best_score_, 3))
    print("Test accuracy:", round(accuracy_score(y_test, rf_predictions), 3))
    print(
        "Test lapsed-class F1:",
        round(f1_score(y_test, rf_predictions), 3),
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, rf_predictions))
    print("\nClassification report:")
    print(classification_report(y_test, rf_predictions, zero_division=0))
    
    # Oversample lapsed customers in the training set only
    training_data = X_train.copy()
    training_data["lapsed"] = y_train.to_numpy()

    lapsed_rows = training_data[
        training_data["lapsed"] == 1
    ]

    not_lapsed_rows = training_data[
        training_data["lapsed"] == 0
    ]

    extra_lapsed = lapsed_rows.sample(
        n=len(not_lapsed_rows) - len(lapsed_rows),
        replace=True,
        random_state=42,
    )

    balanced_training = pd.concat(
        [training_data, extra_lapsed],
        ignore_index=True,
    )

    X_balanced = balanced_training.drop(columns=["lapsed"])
    y_balanced = balanced_training["lapsed"]

    print("\nOVERSAMPLING CHECK")
    print("=" * 40)
    print("Original training counts:")
    print(y_train.value_counts().sort_index().to_dict())
    print("Balanced training counts:")
    print(y_balanced.value_counts().sort_index().to_dict())

    # Clone the tuned Logistic Regression pipeline
    balanced_model = clone(search.best_estimator_)
    balanced_model.fit(X_balanced, y_balanced)

    balanced_predictions = balanced_model.predict(X_test)

    print("\nLOGISTIC REGRESSION AFTER OVERSAMPLING")
    print("=" * 40)
    print(
        "Test accuracy:",
        round(accuracy_score(y_test, balanced_predictions), 3),
    )
    print(
        "Lapsed-class F1:",
        round(f1_score(y_test, balanced_predictions), 3),
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, balanced_predictions))
    print(
        classification_report(
            y_test,
            balanced_predictions,
            zero_division=0,
        )
    )
    
    # Save the oversampled Logistic Regression model
    model_path = PROJECT_ROOT / "outputs" / "lapse_model.joblib"
    joblib.dump(balanced_model, model_path)

    print("\nSaved selected model to:", model_path)

    # Reload and verify predictions
    loaded_model = joblib.load(model_path)
    reloaded_predictions = loaded_model.predict(X_test)

    predictions_match = (
        balanced_predictions == reloaded_predictions
    ).all()

    print("Reloaded model predictions match:", predictions_match)
    
    # Build a retention list for Premium customers
    segments_path = PROJECT_ROOT / "outputs" / "customer_segments.csv"
    segments = pd.read_csv(segments_path)

    premium_customers = segments[
        segments["is_premium"] == 1
    ][["customer_id"]].copy()

    # Score the premium customers using their historical features
    premium_features = dataset.merge(
        premium_customers,
        on="customer_id",
        how="inner",
    )

    premium_X = premium_features[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]

    premium_probabilities = balanced_model.predict_proba(
        premium_X
    )[:, 1]

    retention_list = pd.DataFrame({
        "customer_id": premium_features["customer_id"],
        "lapse_probability": premium_probabilities,
        "actually_lapsed": premium_features["lapsed"],
    })

    retention_list = retention_list.sort_values(
        "lapse_probability",
        ascending=False,
    )

    retention_path = PROJECT_ROOT / "outputs" / "retention_list.csv"
    retention_list.to_csv(retention_path, index=False)

    print("\nPREMIUM CUSTOMER RETENTION LIST")
    print("=" * 40)
    print("Premium customers scored:", len(retention_list))
    print(
        "Actually lapsed among Premium:",
        int(retention_list["actually_lapsed"].sum()),
    )
    print("Top 10 customers by lapse probability:")
    print(retention_list.head(10).to_string(index=False))
    print("Saved retention list to:", retention_path)






if __name__ == "__main__":
    main()
