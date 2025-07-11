# tests/test_handlers.py
import pandas as pd
import pytest
from rag_math_engine.core import (
    average_loan_graduates_handler,
    approval_rate_self_employed_handler,
    gender_wise_approval_rate_handler,
    generate_answer_smart
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Education": ["Graduate", "Graduate", "Not Graduate"],
        "LoanAmount": [150, 250, 100],
        "Self_Employed": ["Yes", "No", "Yes"],
        "Loan_Status": ["Y", "N", "Y"],
        "Gender": ["Male", "Female", "Male"],
        "ApplicantIncome": [5000, 3000, 4000],
        "Dependents": ["0", "3+", "1"],
        "Property_Area": ["Urban", "Rural", "Urban"]
    })

def test_average_loan_graduates(sample_df):
    result = average_loan_graduates_handler(sample_df)
    assert "average loan amount for graduates" in result.lower()
    assert "200.00" in result  # (150+250)/2

def test_approval_rate_self_employed(sample_df):
    result = approval_rate_self_employed_handler(sample_df)
    assert "approval rate for self-employed" in result.lower()
    assert "50.00%" in result  # 1 out of 2

def test_gender_wise_approval_rate(sample_df):
    result = gender_wise_approval_rate_handler(sample_df)
    assert "Male: 100.00%" in result
    assert "Female: 0.00%" in result

def test_generate_answer_fallback(sample_df):
    response = generate_answer_smart(sample_df, "average loan graduates")
    assert "average loan amount" in response.lower()
