import os
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(layout="wide")

DB_PATH = "data/predictions.db"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/credit-score")

_LABEL_MAP = {
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

_EXPLAIN = {
    "Credit Utilization": lambda v: f"Credit utilization at {v:.0%} — {'dangerously high' if v > 0.7 else 'elevated' if v > 0.4 else 'acceptable'}. High utilization signals financial stress.",
    "90+ Days Late":       lambda v: f"{'No serious delinquencies' if v == 0 else str(int(v)) + ' serious late payment(s) on record'}. {'Clean history.' if v == 0 else 'Major negative signal for repayment capacity.'}",
    "30-59 Days Late":     lambda v: f"{'No minor delinquencies' if v == 0 else str(int(v)) + ' minor late payment(s) detected'}. {'Positive signal.' if v == 0 else 'Indicates occasional payment stress.'}",
    "FOIR Ratio":          lambda v: f"FOIR at {v:.0%} — {'exceeds RBI guideline of 50%.' if v >= 0.5 else 'within RBI guideline of 50%.'} {'Limited repayment capacity.' if v >= 0.5 else 'Adequate repayment capacity.'}",
    "Monthly Income":      lambda v: f"Monthly income ₹{int(v):,} — {'below threshold for standard products.' if v < 3000 else 'adequate for loan servicing.'}",
    "Applicant Age":       lambda v: f"Age {int(v)} — {'young applicant, limited credit history likely.' if v < 28 else 'mature applicant, stable profile expected.'}",
    "Active Loans":        lambda v: f"{int(v)} active loan(s) — {'over-leveraged.' if v > 7 else 'manageable obligation count.'}",
    "Real Estate Loans":   lambda v: f"{int(v)} real estate loan(s) — {'asset-backed obligations present.' if v > 0 else 'no property-backed obligations.'}",
    "60-89 Days Late":     lambda v: f"{'No moderate delinquencies.' if v == 0 else str(int(v)) + ' moderate late payment(s).'}",
    "Dependents":          lambda v: f"{int(v)} dependent(s) — {'higher financial obligations.' if v > 2 else 'manageable family obligations.'}",
}


def score_label(score):
    if score > 650:
        return "🟢", "Low Risk Score"
    if score >= 500:
        return "🟡", "Medium Risk Score"
    return "🔴", "High Risk Score"


def page_loan_officer():
    # Section 1 — Title
    st.title("Credit Risk Assessment — Loan Officer Portal")

    # Section 2 — Form (left) + Results (right) side by side
    form_col, result_col = st.columns([1, 1])

    with form_col:
        with st.form("loan_form"):
            util = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.5)
            foir = st.slider("FOIR - Fixed Obligation to Income Ratio", 0.0, 1.0, 0.3)
            age = st.number_input("Applicant Age", min_value=18, max_value=80, value=35)
            income = st.number_input("Monthly Income (USD)", min_value=0, value=5000)
            late_3059 = st.number_input("30-59 Days Late Payments", min_value=0, max_value=20, value=0)
            late_6089 = st.number_input("60-89 Days Late Payments", min_value=0, max_value=20, value=0)
            late_90 = st.number_input("90+ Days Late Payments", min_value=0, max_value=20, value=0)
            open_loans = st.number_input("Total Active Loans and Credit Lines", min_value=0, value=5)
            real_estate = st.number_input("Real Estate Loans", min_value=0, max_value=10, value=1)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
            submitted = st.form_submit_button("Assess Credit Risk", use_container_width=True)

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
        try:
            with st.spinner("🔄 Scoring applicant... (first request may take ~30s to wake the server)"):
                response = requests.post(API_URL, json=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to the API server. Please check that the backend is running.")
            st.stop()
        except requests.exceptions.Timeout:
            st.warning("⏳ The API server is warming up (Render free tier cold start). Please wait 30–60 seconds and click **Assess Credit Risk** again.")
            st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ API request failed: {e}")
            st.stop()

        score = result["credit_score"]
        prob = result["default_probability"]
        band = result["risk_band"]
        top3 = result["top_3_reasons"]
        dot, label = score_label(score)

        _input_map = {
            "Credit Utilization": util, "FOIR Ratio": foir, "Applicant Age": age,
            "Monthly Income": income, "30-59 Days Late": late_3059, "60-89 Days Late": late_6089,
            "90+ Days Late": late_90, "Active Loans": open_loans,
            "Real Estate Loans": real_estate, "Dependents": dependents,
        }

        # Right column — Assessment Result
        with result_col:
            st.subheader("Assessment Result")
            c1, c2, c3 = st.columns(3)
            c1.metric("Credit Score", f"{dot} {score}", label)
            c2.metric("Risk Band", band)
            c3.metric("Default Probability", f"{prob * 100:.2f}%")
            if foir < 0.50:
                st.success(f"✅ FOIR: {foir:.2f} — Within RBI guideline (max 0.50)")
            else:
                st.warning(f"⚠️ FOIR: {foir:.2f} — Exceeds RBI guideline (max 0.50)")
            _labels = [_LABEL_MAP.get(r["feature"], r["feature"]) for r in top3]
            _values = [r["shap_value"] for r in top3]
            _colors = ["#FF4B4B" if v > 0 else "#00CC44" for v in _values]
            plt.rcParams["figure.facecolor"] = "#0e1117"
            plt.rcParams["axes.facecolor"] = "#0e1117"
            plt.rcParams["text.color"] = "white"
            plt.rcParams["axes.labelcolor"] = "white"
            plt.rcParams["xtick.color"] = "white"
            fig_shap, ax_shap = plt.subplots(figsize=(6, 3))
            ax_shap.barh(_labels, _values, color=_colors)
            ax_shap.axvline(0, color="white", linewidth=0.8)
            ax_shap.set_xlabel("SHAP Value")
            st.pyplot(fig_shap)

        # Section 3 — Decision Explanation full width
        st.divider()
        st.subheader("Decision Explanation")
        exp_left, exp_right = st.columns([1, 1])

        with exp_left:
            if band == "Low Risk":
                st.success("✅ Applicant appears financially stable. Loan recommended for approval.")
            elif band == "Medium Risk":
                st.warning("⚠️ Applicant shows moderate risk. Consider reduced loan amount or higher interest rate.")
            else:
                st.error("🚨 High likelihood of default. Loan not recommended without collateral.")
            st.markdown("<span style='font-size:0.85rem;opacity:0.75;'>📋 RBI Compliance: WoE/IV Logistic Regression | Decision basis: Income, repayment history, credit utilization | Audit trail: Logged to predictions.db</span>", unsafe_allow_html=True)

        with exp_right:
            st.markdown("<p style='font-size:1.1rem;font-weight:700;margin-bottom:0.5rem;'>Key Risk Factors:</p>", unsafe_allow_html=True)
            for r in top3:
                feat_label = _LABEL_MAP.get(r["feature"], r["feature"])
                val = _input_map.get(feat_label, 0)
                explain_fn = _EXPLAIN.get(feat_label)
                explanation = explain_fn(val) if explain_fn else feat_label
                prefix = "⚠️" if r["shap_value"] > 0 else "✅"
                st.markdown(f"<p style='font-size:1.1rem;line-height:1.6;margin:0.4rem 0;'>{prefix} {explanation}</p>", unsafe_allow_html=True)


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
