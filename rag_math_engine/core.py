# core.py - Main math handler logic and dispatcher
import re
from .patterns import detect_math_query_type

# ------------------------
# Dispatchable Math Handlers
# ------------------------

def average_loan_graduates_handler(df, question=None):
    grad_df = df[df["Education"] == "Graduate"]
    avg = grad_df["LoanAmount"].mean()
    return f"The average loan amount for graduates is **{avg:.2f}**."

def approval_rate_self_employed_handler(df, question=None):
    se_df = df[df["Self_Employed"] == "Yes"]
    approved = se_df[se_df["Loan_Status"] == "Y"]
    rate = (approved.shape[0] / se_df.shape[0]) * 100 if not se_df.empty else 0
    return f"The approval rate for self-employed applicants is **{rate:.2f}%**."

def count_approved_loans_handler(df, question=None):
    count = df[df["Loan_Status"] == "Y"].shape[0]
    return f"Total number of approved loans: **{count}**."

def max_applicant_income_handler(df, question=None):
    max_income = df["ApplicantIncome"].max()
    return f"The maximum applicant income is **{max_income}**."

def median_loan_amount_handler(df, question=None):
    median = df["LoanAmount"].median()
    return f"The median loan amount is **{median:.2f}**."

def gender_wise_approval_rate_handler(df, question=None):
    response = []
    for gender in ["Male", "Female"]:
        group = df[df["Gender"] == gender]
        approved = group[group["Loan_Status"] == "Y"]
        rate = (approved.shape[0] / group.shape[0]) * 100 if not group.empty else 0
        response.append(f"{gender}: {rate:.2f}%")
    return "**Gender-wise loan approval rates:**\n" + "\n".join(response)

def education_approval_count_handler(df, question=None):
    grads = df[(df["Education"] == "Graduate") & (df["Loan_Status"] == "Y")].shape[0]
    nongrads = df[(df["Education"] == "Not Graduate") & (df["Loan_Status"] == "Y")].shape[0]
    return f"Graduates approved: **{grads}**\nNon-graduates approved: **{nongrads}**."

def avg_income_by_property_area_handler(df, question=None):
    result = df.groupby("Property_Area")["ApplicantIncome"].mean().round(2)
    response = "**Average Income by Property Area:**\n"
    for area, avg_income in result.items():
        response += f"{area}: ₹{avg_income}\n"
    return response.strip()

def applicants_with_more_dependents_handler(df, question=None):
    count = df[df["Dependents"].isin(["3+", 3])].shape[0]
    return f"Applicants with 3 or more dependents: **{count}**."

def avg_loan_high_income_handler(df, question=None):
    high_income_df = df[df["ApplicantIncome"] > 10000]
    avg = high_income_df["LoanAmount"].mean()
    return f"Average loan amount for applicants with income > ₹10,000: **{avg:.2f}**."


# ------------------------
# Dispatcher
# ------------------------

def dispatch_math_query(qtype, df, question):
    dispatcher = {
        "average_loan_for_graduates": average_loan_graduates_handler,
        "approval_rate_self_employed": approval_rate_self_employed_handler,
        "count_approved_loans": count_approved_loans_handler,
        "max_applicant_income": max_applicant_income_handler,
        "median_loan_amount": median_loan_amount_handler,
        "gender_wise_approval_rate": gender_wise_approval_rate_handler,
        "education_approval_count": education_approval_count_handler,
        "avg_income_by_property_area": avg_income_by_property_area_handler,
        "applicants_with_more_dependents": applicants_with_more_dependents_handler,
        "avg_loan_high_income": avg_loan_high_income_handler
    }
    if qtype in dispatcher:
        return dispatcher[qtype](df, question)
    return None


# ------------------------
# Entry Point
# ------------------------

def generate_answer_smart(df, question, retrieve_similar_docs=None, generate_answer_llm=None):
    qtype = detect_math_query_type(question)
    if qtype:
        return dispatch_math_query(qtype, df, question)
    elif retrieve_similar_docs and generate_answer_llm:
        context = "\n".join(retrieve_similar_docs(question))
        return generate_answer_llm(context, question)
    else:
        return "I'm unable to process this query."
