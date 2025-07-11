# patterns.py
import re
import spacy

# Load spaCy model once	nlp = spacy.load("en_core_web_sm")

# ------------------------
# Regex Rule-Based Classification
# ------------------------

def detect_math_query_type_regex(question):
    q = question.lower()
    patterns = {
        "average_loan_for_graduates": r"average loan amount.*graduate",
        "approval_rate_self_employed": r"approval rate.*self.?employed",
        "count_approved_loans": r"how many loans.*approved",
        "max_applicant_income": r"maximum applicant income",
        "median_loan_amount": r"median loan amount",
        "gender_wise_approval_rate": r"approval rate.*(male|female|gender)",
        "education_approval_count": r"how many.*(graduates|non.?graduates).*approved",
        "avg_income_by_property_area": r"average income.*property area",
        "applicants_with_more_dependents": r"(how many|count).*dependents.*(3|more|greater)",
        "avg_loan_high_income": r"average loan amount.*income.*(10|10000)"
    }
    for qtype, pattern in patterns.items():
        if re.search(pattern, q):
            return qtype
    return None

# ------------------------
# NLP Fallback with spaCy
# ------------------------

def detect_math_query_type_spacy(question):
    doc = nlp(question.lower())
    if "graduate" in question and "average" in question:
        return "average_loan_for_graduates"
    if "income" in question and any(ent.label_ == "MONEY" for ent in doc.ents):
        return "avg_loan_high_income"
    if "approval rate" in question:
        return "gender_wise_approval_rate"
    return None

# ------------------------
# Combined Detection
# ------------------------

def detect_math_query_type(question):
    qtype = detect_math_query_type_regex(question)
    if qtype:
        return qtype
    return detect_math_query_type_spacy(question)
