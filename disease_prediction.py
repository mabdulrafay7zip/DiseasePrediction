"""
Disease Prediction from Medical Data — Machine Learning Project
Author: Muhammad Abdul Rafay — ML Intern

Predicts whether a patient has diabetes using the Pima Indians Diabetes
dataset (medical measurements: glucose, blood pressure, BMI, age, ...).

Models compared:
    1. Logistic Regression
    2. Random Forest Classifier
    3. Support Vector Machine (SVM)

Run:
    pip install -r requirements.txt
    python disease_prediction.py
Outputs:
    outputs/metrics.json
    outputs/confusion_matrix_<model>.png
    outputs/model_comparison.png
    outputs/feature_distributions.png
    data/pima_diabetes.csv   - cached copy of the dataset (downloaded on first run)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = DATA_DIR / "pima_diabetes.csv"
SOURCE_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
# In this dataset a 0 in these columns is physiologically impossible -> missing value.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_dataset() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        print(f"Loading cached dataset: {DATA_FILE}")
        return pd.read_csv(DATA_FILE)

    print(f"Downloading Pima Indians Diabetes dataset...\n  {SOURCE_URL}")
    try:
        df = pd.read_csv(SOURCE_URL, header=None, names=COLUMNS)
    except Exception as exc:  # pragma: no cover - network fallback
        raise RuntimeError(
            f"Could not download the dataset ({exc}). Download it from {SOURCE_URL} "
            f"and save it as {DATA_FILE} with columns {COLUMNS}."
        ) from exc
    df.to_csv(DATA_FILE, index=False)
    print(f"Saved cached copy to {DATA_FILE}")
    return df


def evaluate(name: str, y_true, y_pred) -> dict:
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def save_confusion_matrix(name: str, y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["No diabetes", "Diabetes"],
                yticklabels=["No diabetes", "Diabetes"])
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def save_feature_distributions(df: pd.DataFrame) -> None:
    melted = df.copy()
    melted[ZERO_AS_MISSING] = melted[ZERO_AS_MISSING].replace(0, float("nan"))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.ravel(), COLUMNS[:-1]):
        sns.histplot(data=melted, x=col, hue="Outcome", bins=25, ax=ax, legend=False)
        ax.set_title(col)
    fig.suptitle("Feature distributions by Outcome (0 = no diabetes, 1 = diabetes)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Outcome distribution: {df['Outcome'].value_counts().to_dict()}  (1 = diabetes)\n")

    save_feature_distributions(df)

    X = df.drop(columns=["Outcome"]).copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, float("nan"))
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True),
    }

    results = []
    for name, model in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", model),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append(evaluate(name, y_test, pred))
        save_confusion_matrix(name, y_test, pred)
        print(f"--- {name} ---")
        print(classification_report(y_test, pred, target_names=["No diabetes", "Diabetes"]))

    metrics = pd.DataFrame(results)
    metrics.set_index("model").plot(kind="bar", figsize=(9, 5), ylim=(0, 1))
    plt.title("Diabetes Prediction — Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close()

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== Summary ===")
    print(metrics.to_string(index=False))
    print(f"\nPlots and metrics saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
