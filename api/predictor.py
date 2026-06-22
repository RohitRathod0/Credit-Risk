import pickle
import numpy as np
import pandas as pd
import shap
from datetime import datetime, timezone

with open("models/woe_pipeline.pkl", "rb") as f:
    woe_pipeline = pickle.load(f)

with open("models/logreg_model.pkl", "rb") as f:
    logreg_model = pickle.load(f)

MODEL_VERSION = "logreg_v1"

_background = np.zeros((1, len(logreg_model.coef_[0])))
_explainer = shap.LinearExplainer(logreg_model, _background)


def _risk_band(prob: float) -> str:
    if prob < 0.2:
        return "Low Risk"
    if prob <= 0.5:
        return "Medium Risk"
    return "High Risk"


def score_applicant(data: dict) -> dict:
    df = pd.DataFrame([data])
    woe_df = woe_pipeline.transform(df)
    prob = float(logreg_model.predict_proba(woe_df)[0][1])
    raw_shap = _explainer.shap_values(woe_df)
    sv = (raw_shap[1] if isinstance(raw_shap, list) else raw_shap)[0]
    cols = list(woe_df.columns) if hasattr(woe_df, "columns") else [f"f{i}" for i in range(len(sv))]
    ranked = sorted(zip(cols, sv), key=lambda x: abs(x[1]), reverse=True)
    top_3 = [{"feature": f, "shap_value": round(float(v), 4)} for f, v in ranked[:3]]
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
