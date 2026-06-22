import joblib
import numpy as np
import pandas as pd
import shap
from datetime import datetime, timezone
from sklearn import set_config

set_config(transform_output="pandas")

woe_pipeline = joblib.load("models/woe_pipeline.pkl")
logreg_model = joblib.load("models/logreg_model.pkl")

MODEL_VERSION = "logreg_v1"

_imputer = woe_pipeline.named_steps["imputer"]
_woe = woe_pipeline.named_steps["woe"]

COLUMN_MAP = {
    "NumberOfTime3059DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime6089DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse",
}

# Original input field names in column order (index matches fN from WOE transformer)
FEATURE_NAMES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime3059DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime6089DaysPastDueNotWorse",
    "NumberOfDependents",
]

_background = np.zeros((1, len(logreg_model.coef_[0])))
_explainer = shap.LinearExplainer(logreg_model, _background)


def _risk_band(prob: float) -> str:
    if prob < 0.2:
        return "Low Risk"
    if prob <= 0.5:
        return "Medium Risk"
    return "High Risk"


def score_applicant(data: dict) -> dict:
    renamed = {COLUMN_MAP.get(k, k): v for k, v in data.items()}
    df = pd.DataFrame([renamed])
    X_imp = _imputer.transform(df)
    woe_df = _woe.transform(X_imp)
    prob = float(logreg_model.predict_proba(woe_df)[0][1])
    raw_shap = _explainer.shap_values(woe_df)
    sv = (raw_shap[1] if isinstance(raw_shap, list) else raw_shap)[0]
    cols = list(woe_df.columns) if hasattr(woe_df, "columns") else [f"f{i}" for i in range(len(sv))]
    ranked = sorted(zip(cols, sv), key=lambda x: abs(x[1]), reverse=True)
    def _readable_name(col: str) -> str:
        """Map internal fN column names back to original input field names."""
        if col.startswith("f") and col[1:].isdigit():
            idx = int(col[1:])
            if idx < len(FEATURE_NAMES):
                return FEATURE_NAMES[idx]
        return col
    top_3 = [{"feature": _readable_name(f), "shap_value": round(float(v), 4)} for f, v in ranked[:3]]
    return {
        "default_probability": round(prob, 4),
        "risk_band": _risk_band(prob),
        "credit_score": int(850 - (prob * 550)),
        "top_3_reasons": top_3,
        "model_version": MODEL_VERSION,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def score_batch(records: list) -> list:
    return [score_applicant(r) for r in records]
