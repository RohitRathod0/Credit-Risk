import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

DB_PATH = "data/predictions.db"
API_URL = "http://127.0.0.1:8000/credit-score"


def score_label(score):
    if score > 650:
        return "🟢", "Low Risk Score"
    if score >= 500:
        return "🟡", "Medium Risk Score"
    return "🔴", "High Risk Score"


def page_loan_officer():
    st.title("Credit Risk Assessment — Loan Officer Portal")

    with st.form("loan_form"):
        util = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.5)
        age = st.number_input("Applicant Age", min_value=18, max_value=80, value=35)
        late_3059 = st.number_input("30-59 Days Late Payments", min_value=0, max_value=20, value=0)
        foir = st.slider("FOIR - Fixed Obligation to Income Ratio", 0.0, 1.0, 0.3)
        income = st.number_input("Monthly Income (USD)", min_value=0, value=5000)
        open_loans = st.number_input("Total Active Loans and Credit Lines", min_value=0, value=5)
        late_90 = st.number_input("90+ Days Late Payments", min_value=0, max_value=20, value=0)
        real_estate = st.number_input("Real Estate Loans", min_value=0, max_value=10, value=1)
        late_6089 = st.number_input("60-89 Days Late Payments", min_value=0, max_value=20, value=0)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
        submitted = st.form_submit_button("Assess Credit Risk")

    if submitted:
        payload = {
            "RevolvingUtilizationOfUnsecuredLines": util,
            "age": int(age),
            "NumberOfTime3059DaysPastDueNotWorse": int(late_3059),
            "DebtRatio": foir,
            "MonthlyIncome": float(income),
            "NumberOfOpenCreditLinesAndLoans": int(open_loans),
            "NumberOfTimes90DaysLate": int(late_90),
            "NumberRealEstateLoansOrLines": int(real_estate),
            "NumberOfTime6089DaysPastDueNotWorse": int(late_6089),
            "NumberOfDependents": float(dependents),
        }
        result = requests.post(API_URL, json=payload).json()

        score = result["credit_score"]
        prob = result["default_probability"]
        band = result["risk_band"]
        top3 = result["top_3_reasons"]
        dot, label = score_label(score)

        c1, c2, c3 = st.columns(3)
        c1.metric("Credit Score", f"{dot} {score}", label)
        c2.metric("Risk Band", band)
        c3.metric("Default Probability", f"{prob * 100:.2f}%")

        st.subheader("Top 3 Factors Affecting This Decision")
        _label_map = {
            "RevolvingUtilizationOfUnsecuredLines": "Credit Utilization",
            "NumberOfTimes90DaysLate": "90+ Days Late",
            "NumberOfTime3059DaysPastDueNotWorse": "30-59 Days Late",
            "DebtRatio": "FOIR Ratio",
            "MonthlyIncome": "Monthly Income",
            "age": "Applicant Age",
            "NumberOfOpenCreditLinesAndLoans": "Active Loans",
            "NumberRealEstateLoansOrLines": "Real Estate Loans",
            "NumberOfTime6089DaysPastDueNotWorse": "60-89 Days Late",
            "NumberOfDependents": "Dependents",
        }
        _labels = [_label_map.get(r["feature"], r["feature"]) for r in top3]
        _values = [r["shap_value"] for r in top3]
        _colors = ["#FF4B4B" if v > 0 else "#00CC44" for v in _values]
        fig_shap, ax_shap = plt.subplots()
        ax_shap.barh(_labels, _values, color=_colors)
        ax_shap.axvline(0, color="white", linewidth=0.8)
        ax_shap.set_xlabel("SHAP Value")
        st.pyplot(fig_shap)


def page_portfolio():
    st.title("Portfolio Risk Dashboard")

    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM predictions", con)
    con.close()

    df["scored_at"] = pd.to_datetime(df["scored_at"], utc=True)
    today = pd.Timestamp.now(tz="UTC").normalize()
    today_df = df[df["scored_at"] >= today]

    avg_score = round(df["credit_score"].mean(), 1) if len(df) else 0
    approval_rate = round((df["risk_band"] == "Low Risk").mean() * 100, 1) if len(df) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications Today", len(today_df))
    c2.metric("Avg Credit Score", avg_score)
    c3.metric("Approval Rate", f"{approval_rate}%")
    c4.metric("Total Scored", len(df))

    st.subheader("Risk Band Distribution")
    band_counts = df["risk_band"].value_counts()
    _pie_color_map = {"High Risk": "#FF4B4B", "Medium Risk": "#FFA500", "Low Risk": "#00CC44"}
    _pie_colors = [_pie_color_map.get(b, "#888888") for b in band_counts.index]
    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.pie(band_counts.values, labels=band_counts.index, autopct="%1.1f%%", colors=_pie_colors)
    st.pyplot(fig)

    st.subheader("Last 10 Scored Applications")
    display_cols = ["scored_at", "credit_score", "risk_band", "default_probability"]
    st.dataframe(df.sort_values("scored_at", ascending=False)[display_cols].head(10))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Export as CSV", csv, "predictions.csv", "text/csv")


page = st.sidebar.radio("Navigation", ["Loan Officer View", "Risk Manager Portfolio View"])
if page == "Loan Officer View":
    page_loan_officer()
else:
    page_portfolio()
