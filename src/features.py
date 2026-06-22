import pandas as pd
import numpy as np
import pickle
from optbinning import OptimalBinning

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
IV_THRESHOLD = 0.02


def load_data(path="data/cs-training.csv"):
    return pd.read_csv(path, index_col=0)


def impute(df):
    df = df.copy()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].mode()[0]
    )
    return df


def fit_woe_binners(df, features, target):
    binners = {}
    iv_records = []
    for feat in features:
        x = df[feat].values
        y = df[target].values
        ob = OptimalBinning(name=feat, dtype="numerical", solver="cp")
        ob.fit(x, y)
        ob.binning_table.build()
        binners[feat] = ob
        iv = ob.binning_table.iv
        iv_records.append({"feature": feat, "IV": round(float(iv), 4)})
    return binners, iv_records


def apply_woe(df, binners):
    df = df.copy()
    for feat, ob in binners.items():
        df[f"{feat}_woe"] = ob.transform(df[feat].values, metric="woe")
    return df


def save_iv_report(iv_records, path="reports/iv_report.csv"):
    import os
    os.makedirs("reports", exist_ok=True)
    iv_df = pd.DataFrame(iv_records).sort_values("IV", ascending=False)
    iv_df["keep"] = iv_df["IV"] >= IV_THRESHOLD
    iv_df.to_csv(path, index=False)
    return iv_df


def get_selected_features(iv_df):
    kept = iv_df[iv_df["keep"]]["feature"].tolist()
    return [f"{f}_woe" for f in kept]


def main():
    df = load_data()
    df = impute(df)

    binners, iv_records = fit_woe_binners(df, FEATURES, TARGET)

    iv_df = save_iv_report(iv_records)
    print("\nIV Report:")
    print(iv_df.to_string(index=False))

    df = apply_woe(df, binners)
    selected = get_selected_features(iv_df)
    print(f"\nSelected WoE features ({len(selected)}): {selected}")

    import os
    os.makedirs("models", exist_ok=True)
    with open("models/woe_binners.pkl", "wb") as f:
        pickle.dump(binners, f)

    df.to_csv("data/cs-training-woe.csv", index=True)
    print("\nSaved: models/woe_binners.pkl | data/cs-training-woe.csv | reports/iv_report.csv")


if __name__ == "__main__":
    main()
