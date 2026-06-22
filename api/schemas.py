from pydantic import BaseModel
from typing import List, Dict, Any


class CreditInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime3059DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime6089DaysPastDueNotWorse: int
    NumberOfDependents: float


class CreditScoreResponse(BaseModel):
    default_probability: float
    risk_band: str
    credit_score: int
    top_3_reasons: List[Dict[str, Any]]
    model_version: str
    scored_at: str
