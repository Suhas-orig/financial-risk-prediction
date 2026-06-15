import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Load Model & Scaler
# ----------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Financial Risk Predictor",
    page_icon="💰",
    layout="wide"
)

st.title(" Financial Risk Prediction System")
st.write("Random Forest based Financial Risk Prediction")

st.sidebar.header("Project Information")
st.sidebar.write("Model Used: Random Forest")
st.sidebar.write("Dataset: Personal Finance Tracker")
st.sidebar.success("Best Accuracy: 94.17%")

st.markdown("---")

col1, col2 = st.columns(2)

# ----------------------------
# Left Column
# ----------------------------

with col1:

    monthly_income = st.number_input("Monthly Income", value=50000.0)

    monthly_expense_total = st.number_input("Monthly Expense", value=25000.0)

    savings_rate = st.slider(
        "Savings Rate",
        0.0,
        1.0,
        0.30
    )

    budget_goal = st.number_input("Budget Goal", value=40000.0)

    financial_scenario = st.selectbox(
        "Financial Scenario",
        [
            "Poor",
            "Average",
            "Good"
        ]
    )

    financial_scenario = {
        "Poor": 0,
        "Average": 1,
        "Good": 2
    }[financial_scenario]

    credit_score = st.slider(
        "Credit Score",
        300,
        900,
        700
    )

    debt_to_income_ratio = st.slider(
        "Debt To Income Ratio",
        0.0,
        1.0,
        0.20
    )

    loan_payment = st.number_input(
        "Loan Payment",
        value=5000.0
    )

    investment_amount = st.number_input(
        "Investment Amount",
        value=10000.0
    )

    subscription_services = st.selectbox(
        "Subscription Services",
        [
            "No",
            "Yes"
        ]
    )

    subscription_services = {
        "No": 0,
        "Yes": 1
    }[subscription_services]

    emergency_fund = st.number_input(
        "Emergency Fund",
        value=50000.0
    )

    transaction_count = st.number_input(
        "Transaction Count",
        value=120
    )

# ----------------------------
# Right Column
# ----------------------------

with col2:

    fraud_flag = st.selectbox(
        "Previous Fraud Detected",
        [
            "No",
            "Yes"
        ]
    )

    fraud_flag = {
        "No": 0,
        "Yes": 1
    }[fraud_flag]

    discretionary_spending = st.number_input(
        "Discretionary Spending",
        value=8000.0
    )

    essential_spending = st.number_input(
        "Essential Spending",
        value=20000.0
    )

    income_type = st.selectbox(
        "Income Type",
        [
            "Freelance",
            "Mixed",
            "Salary"
        ]
    )

    income_type = {
        "Freelance": 0,
        "Mixed": 1,
        "Salary": 2
    }[income_type]

    rent_or_mortgage = st.number_input(
        "Rent / Mortgage",
        value=15000.0
    )

    category = st.selectbox(
        "Category",
        [
            "Dining Out",
            "Education",
            "Entertainment",
            "Groceries",
            "Healthcare",
            "Insurance",
            "Investments",
            "Rent",
            "Transportation",
            "Utilities"
        ]
    )

    category = {
        "Dining Out": 0,
        "Education": 1,
        "Entertainment": 2,
        "Groceries": 3,
        "Healthcare": 4,
        "Insurance": 5,
        "Investments": 6,
        "Rent": 7,
        "Transportation": 8,
        "Utilities": 9
    }[category]

    cash_flow_status = st.selectbox(
        "Cash Flow Status",
        [
            "Negative",
            "Neutral",
            "Positive"
        ]
    )

    cash_flow_status = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }[cash_flow_status]

    financial_advice_score = st.slider(
        "Financial Advice Score",
        0.0,
        10.0,
        5.0
    )

    actual_savings = st.number_input(
        "Actual Savings",
        value=12000.0
    )

    savings_goal_met = st.selectbox(
        "Savings Goal Achieved",
        [
            "No",
            "Yes"
        ]
    )

    savings_goal_met = {
        "No": 0,
        "Yes": 1
    }[savings_goal_met]

# ----------------------------
# Derived Features
# ----------------------------

expense_income_ratio = monthly_expense_total / monthly_income if monthly_income != 0 else 0

savings_amount = monthly_income * savings_rate

# ----------------------------
# Predict
# ----------------------------

if st.button("Predict Risk"):

    data = pd.DataFrame([[
        monthly_income,
        monthly_expense_total,
        savings_rate,
        budget_goal,
        financial_scenario,
        credit_score,
        debt_to_income_ratio,
        loan_payment,
        investment_amount,
        subscription_services,
        emergency_fund,
        transaction_count,
        fraud_flag,
        discretionary_spending,
        essential_spending,
        income_type,
        rent_or_mortgage,
        category,
        cash_flow_status,
        financial_advice_score,
        actual_savings,
        savings_goal_met,
        expense_income_ratio,
        savings_amount
    ]], columns=[
        'monthly_income',
        'monthly_expense_total',
        'savings_rate',
        'budget_goal',
        'financial_scenario',
        'credit_score',
        'debt_to_income_ratio',
        'loan_payment',
        'investment_amount',
        'subscription_services',
        'emergency_fund',
        'transaction_count',
        'fraud_flag',
        'discretionary_spending',
        'essential_spending',
        'income_type',
        'rent_or_mortgage',
        'category',
        'cash_flow_status',
        'financial_advice_score',
        'actual_savings',
        'savings_goal_met',
        'expense_income_ratio',
        'savings_amount'
    ])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    st.markdown("---")
    st.subheader("Prediction")

    risk = prediction[0]

    if risk == 0:
        st.success("""
    ### 🟢 Low Financial Risk

    The user's financial profile appears healthy with a low level of financial risk.
    """)

    elif risk == 1:
        st.warning("""
    ### 🟠 Medium Financial Risk

    The user has moderate financial risk. Some financial indicators may require attention.
    """)

    else:
        st.error("""
    ### 🔴 High Financial Risk

    The user's financial profile indicates a high level of financial risk. Financial planning and corrective actions are recommended.
    """)