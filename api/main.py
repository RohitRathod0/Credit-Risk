import json
import os
import sqlite3
from fastapi import FastAPI
from api.schemas import CreditInput, CreditScoreResponse
from api.predictor import score_applicant, score_batch, MODEL_VERSION

app = FastAPI()
DB_PATH = "data/predictions.db"

os.makedirs("data", exist_ok=True)
con = sqlite3.connect(DB_PATH)
con.execute(
    "CREATE TABLE IF NOT EXISTS predictions "
    "(id INTEGER PRIMARY KEY AUTOINCREMENT, input_json TEXT, default_probability REAL, "
    "credit_score INTEGER, risk_band TEXT, model_version TEXT, scored_at TEXT)"
)
con.commit()
con.close()


def _log(data: dict, result: dict):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO predictions (input_json, default_probability, credit_score, risk_band, model_version, scored_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (json.dumps(data), result["default_probability"], result["credit_score"],
         result["risk_band"], result["model_version"], result["scored_at"]),
    )
    con.commit()
    con.close()


@app.post("/credit-score", response_model=CreditScoreResponse)
def credit_score(payload: CreditInput):
    data = payload.model_dump()
    result = score_applicant(data)
    _log(data, result)
    return result


@app.post("/batch-score")
def batch_score(payload: list[CreditInput]):
    records = [p.model_dump() for p in payload]
    results = score_batch(records)
    for data, result in zip(records, results):
        _log(data, result)
    return results


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}
