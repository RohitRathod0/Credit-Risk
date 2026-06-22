import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics import roc_auc_score
from sklearn import set_config

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
TARGET = "SeriousDlqin2yrs"
WOE_FEATURES = [f"{f}_woe" for f in FEATURES]


def gini(y_true, y_prob):
    return 2 * roc_auc_score(y_true, y_prob) - 1


def ks_stat(y_true, y_prob):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    n_pos = y_true.sum()
    n_neg = (1 - y_true).sum()
    df["cum_pos"] = (df["y"] == 1).cumsum() / n_pos
    df["cum_neg"] = (df["y"] == 0).cumsum() / n_neg
    return (df["cum_pos"] - df["cum_neg"]).abs().max()


def load_test_data():
    with open("models/split_indices.pkl", "rb") as f:
        indices = pickle.load(f)

    raw = pd.read_csv("data/cs-training.csv", index_col=0)
    woe_df = pd.read_csv("data/cs-training-woe.csv", index_col=0)

    test_idx = indices["test"]
    X_test_raw = raw.loc[test_idx, FEATURES]
    X_test_woe = woe_df.loc[test_idx, WOE_FEATURES]
    y_test = raw.loc[test_idx, TARGET]
    return X_test_raw, X_test_woe, y_test


def print_metrics(name, y_true, y_prob):
    g = gini(y_true, y_prob)
    ks = ks_stat(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    status_gini = "PASS" if g >= 0.5 else "FAIL (target >= 0.50)"
    status_ks = "PASS" if ks >= 0.35 else "FAIL (target >= 0.35)"
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  AUC     : {auc:.4f}")
    print(f"  Gini    : {g:.4f}  [{status_gini}]")
    print(f"  KS Stat : {ks:.4f}  [{status_ks}]")


def main():
    set_config(transform_output="pandas")
    X_test_raw, X_test_woe, y_test = load_test_data()

    woe_pipeline = joblib.load("models/woe_pipeline.pkl")
    prob_lr = woe_pipeline.predict_proba(X_test_raw)[:, 1]
    print_metrics("Logistic Regression (WoE Pipeline)", y_test, prob_lr)

    lgbm_model = joblib.load("models/lgbm_model.pkl")
    prob_lgbm = lgbm_model.predict_proba(X_test_woe)[:, 1]
    print_metrics("LightGBM Challenger", y_test, prob_lgbm)

    print(f"\n{'='*50}")


if __name__ == "__main__":
    main()
