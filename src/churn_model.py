"""
Churn prediction model.

Approach:
1. Train/test split, STRATIFIED on the label (keeps the ~56/44 class balance
   consistent in both sets — important since we're already close to balanced,
   don't want an unlucky split to skew one set).
2. Baseline: Logistic Regression (scaled features) — simple, interpretable,
   gives us a floor to beat.
3. Stronger model: XGBoost — captures non-linear relationships and feature
   interactions the baseline can't.
4. Evaluation uses precision, recall, F1, and ROC-AUC — NOT just accuracy.
   Even with a roughly balanced label here, accuracy alone doesn't tell us
   which TYPE of mistake the model makes. For a churn model, missing an
   actual churner (false negative -> low recall) is usually more costly than
   wrongly flagging a loyal customer for a retention offer (false positive ->
   low precision), so recall on the churned class matters most operationally.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

FEATURE_COLS = [
    "Recency", "Frequency", "Monetary", "tenure_days",
    "n_unique_products", "avg_order_value", "return_rate",
]


def load_data(path="data/processed/churn_model_dataset.csv"):
    df = pd.read_csv(path, index_col="CustomerID")
    X = df[FEATURE_COLS]
    y = df["is_churned"]
    return X, y


def train_logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model


def evaluate(name, y_test, y_pred, y_proba):
    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print()


if __name__ == "__main__":
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train churn rate: {y_train.mean()*100:.1f}%, Test churn rate: {y_test.mean()*100:.1f}%")
    print()

    # --- Baseline: Logistic Regression ---
    log_reg, scaler = train_logistic_regression(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
    evaluate("Logistic Regression (baseline)", y_test, y_pred_lr, y_proba_lr)

    # --- Stronger model: XGBoost ---
    xgb = train_xgboost(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    evaluate("XGBoost", y_test, y_pred_xgb, y_proba_xgb)

    # --- Feature importance (XGBoost) ---
    importance = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("XGBoost feature importance:")
    print(importance)