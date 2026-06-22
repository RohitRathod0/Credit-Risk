import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from optbinning import BinningProcess
import lightgbm as lgb
import os

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


def load_raw(path="data/cs-training.csv"):
    return pd.read_csv(path, index_col=0)


def load_woe(path="data/cs-training-woe.csv"):
    return pd.read_csv(path, index_col=0)


def make_array_to_df(feature_names):
    def _fn(X):
        return pd.DataFrame(X, columns=feature_names)
    return FunctionTransformer(_fn)


def build_woe_pipeline(features):
    binning_transform_params = {feat: {"metric": "woe"} for feat in features}
    binning_process = BinningProcess(
        variable_names=features,
        binning_transform_params=binning_transform_params,
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("to_df", make_array_to_df(features)),
        ("woe", binning_process),
        ("logreg", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.1,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    return pipeline


def train_lgbm(X_train, y_train):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def main():
    os.makedirs("models", exist_ok=True)

    raw = load_raw()
    X_raw = raw[FEATURES]
    y = raw[TARGET]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting WoE pipeline + LogisticRegression...")
    woe_pipeline = build_woe_pipeline(FEATURES)
    woe_pipeline.fit(X_train_raw, y_train)
    joblib.dump(woe_pipeline, "models/woe_pipeline.pkl")
    logreg = woe_pipeline.named_steps["logreg"]
    joblib.dump(logreg, "models/logreg_model.pkl")
    print("  Saved: models/woe_pipeline.pkl | models/logreg_model.pkl")

    woe_df = load_woe()
    idx_train = X_train_raw.index
    idx_test = X_test_raw.index
    X_train_woe = woe_df.loc[idx_train, WOE_FEATURES]
    X_test_woe = woe_df.loc[idx_test, WOE_FEATURES]

    print("Fitting LightGBM challenger...")
    lgbm_model = train_lgbm(X_train_woe, y_train)
    joblib.dump(lgbm_model, "models/lgbm_model.pkl")
    print("  Saved: models/lgbm_model.pkl")

    indices = {"train": idx_train.tolist(), "test": idx_test.tolist()}
    with open("models/split_indices.pkl", "wb") as f:
        pickle.dump(indices, f)

    print("\nAll models saved. Run src/evaluate.py to check Gini & KS.")


if __name__ == "__main__":
    main()
