import pandas as pd


def main():
    df = pd.read_csv("data/cs-training.csv", index_col=0)

    print("=" * 55)
    print("DATASET SHAPE")
    print(f"  Rows: {df.shape[0]:,}  |  Cols: {df.shape[1]}")

    print("\n" + "=" * 55)
    print("CLASS DISTRIBUTION  (SeriousDlqin2yrs)")
    counts = df["SeriousDlqin2yrs"].value_counts().sort_index()
    total = len(df)
    for label, cnt in counts.items():
        tag = "Default" if label == 1 else "No Default"
        print(f"  {label} ({tag}): {cnt:>7,}  ({cnt/total*100:.2f}%)")

    print("\n" + "=" * 55)
    print("MISSING VALUES")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    for col, cnt in missing.items():
        print(f"  {col:<35} {cnt:>6,}  ({cnt/total*100:.2f}%)")

    print("\n" + "=" * 55)
    print("AGE COLUMN STATS  (checking 0-age outliers)")
    print(f"  Min  : {df['age'].min()}")
    print(f"  Max  : {df['age'].max()}")
    print(f"  Mean : {df['age'].mean():.2f}")
    zero_age = (df["age"] == 0).sum()
    print(f"  age == 0 count: {zero_age}")

    print("\n" + "=" * 55)
    print("REVOLVING UTILIZATION OUTLIERS  (values > 1.0)")
    col = "RevolvingUtilizationOfUnsecuredLines"
    above_one = (df[col] > 1.0).sum()
    print(f"  {col}")
    print(f"  Values > 1.0 : {above_one:,}  ({above_one/total*100:.2f}%)")
    print("=" * 55)


if __name__ == "__main__":
    main()
