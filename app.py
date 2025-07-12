import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import traceback

# Page configuration
st.set_page_config(
    page_title="Loan Query Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0369a1 0%, #0284c7 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .answer-section {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #0369a1;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .error-box {
        background: #fee2e2;
        border: 1px solid #fca5a5;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #dbeafe;
        border: 1px solid #93c5fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Core functions
def load_sample_data():
    """Generate sample loan data for demonstration"""
    np.random.seed(42)
    n_records = 1000
    
    data = {
        'loan_id': [f'LN{str(i).zfill(6)}' for i in range(1, n_records + 1)],
        'borrower_name': [f'Borrower_{i}' for i in range(1, n_records + 1)],
        'loanamount': np.random.normal(50000, 20000, n_records).round(2),
        'interestrate': np.random.normal(7.5, 2.5, n_records).round(2),
        'term_months': np.random.choice([12, 24, 36, 48, 60], n_records),
        'credit_score': np.random.randint(300, 850, n_records),
        'income': np.random.normal(75000, 25000, n_records).round(2),
        'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_records, p=[0.7, 0.25, 0.05]),
        'loan_status': np.random.choice(['Approved', 'Pending', 'Rejected'], n_records, p=[0.6, 0.2, 0.2]),
        'monthly_payment': 0  # Will calculate this
    }
    
    df = pd.DataFrame(data)
    
    # Calculate monthly payment using loan formula
    principal = df['loanamount']
    if "interestrate" in df.columns:
        rate = df["interestrate"] / 100 / 12
    else:
        st.warning("⚠️ This dataset does not contain an 'InterestRate' column.")
        rate = None  # Optional: fallback to zero or default value

    months = df['term_months']
    df['monthly_payment'] = (principal * rate * (1 + rate)**months) / ((1 + rate)**months - 1)
    df['monthly_payment'] = df['monthly_payment'].round(2)
    
    return df

def generate_answer_smart(df: pd.DataFrame, question: str, **kwargs) -> str:
    """Smart answer generation with mathematical computations"""
    question_lower = question.lower()
    
    try:
        # Statistical queries
        if any(word in question_lower for word in ['average', 'mean', 'avg']):
            if 'loan amount' in question_lower:
                avg_amount = df['loanamount'].mean()
                return f"The average loan amount is **${avg_amount:,.2f}**"
            elif 'interestrate' in question_lower:
                avg_rate = df["interestrate"].mean() if "interestrate" in df.columns else None
                return f"The average interest rate is **{avg_rate:.2f}%**"
            elif 'credit score' in question_lower:
                avg_score = df['credit_score'].mean()
                return f"The average credit score is **{avg_score:.0f}**"
        
        # Count queries
        elif any(word in question_lower for word in ['how many', 'count', 'number of']):
            if 'approved' in question_lower:
                approved_count = len(df[df['loan_status'] == 'Approved'])
                total_count = len(df)
                percentage = (approved_count / total_count) * 100
                return f"There are **{approved_count:,}** approved loans out of {total_count:,} total loans (**{percentage:.1f}%**)"
            elif 'rejected' in question_lower:
                rejected_count = len(df[df['loan_status'] == 'Rejected'])
                total_count = len(df)
                percentage = (rejected_count / total_count) * 100
                return f"There are **{rejected_count:,}** rejected loans out of {total_count:,} total loans (**{percentage:.1f}%**)"
        
        # Highest/lowest queries
        elif 'highest' in question_lower or 'maximum' in question_lower:
            if 'loan amount' in question_lower:
                max_amount = df['loanamount'].max()
                return f"The highest loan amount is **${max_amount:,.2f}**"
            elif 'interestrate' in question_lower:
                max_rate = df["interestrate"].max() if "interestrate" in df.columns else None
                return f"The highest interest rate is **{max_rate:.2f}%**"
        
        elif 'lowest' in question_lower or 'minimum' in question_lower:
            if 'loan amount' in question_lower:
                min_amount = df['loanamount'].min()
                return f"The lowest loan amount is **${min_amount:,.2f}**"
            elif 'interest rate' in question_lower:
                min_rate = df["interestrate"].max() if "interestrate" in df.columns else None
                return f"The lowest interest rate is **{min_rate:.2f}%**"
        
        # Credit score analysis
        elif 'credit score' in question_lower:
            if 'distribution' in question_lower:
                excellent = len(df[df['credit_score'] >= 750])
                good = len(df[(df['credit_score'] >= 700) & (df['credit_score'] < 750)])
                fair = len(df[(df['credit_score'] >= 650) & (df['credit_score'] < 700)])
                poor = len(df[df['credit_score'] < 650])
                
                return f"""**Credit Score Distribution:**
                - Excellent (750+): **{excellent:,}** borrowers
                - Good (700-749): **{good:,}** borrowers  
                - Fair (650-699): **{fair:,}** borrowers
                - Poor (<650): **{poor:,}** borrowers"""
        
        # Default general response
        return f"I found **{len(df):,}** loan records in the dataset. Please ask a more specific question about loan amounts, interest rates, credit scores, or loan statuses."
    
    except Exception as e:
        return f"Error processing your question: {str(e)}"

def retrieve_similar_docs(df: pd.DataFrame, question: str) -> List[Dict[str, Any]]:
    """Retrieve relevant loan records based on the question"""
    question_lower = question.lower()
    
    # Simple keyword-based retrieval
    relevant_docs = []
    
    if 'high credit score' in question_lower:
        high_credit = df[df['credit_score'] >= 750].head(5)
        for _, row in high_credit.iterrows():
            relevant_docs.append({
                'loan_id': row['loan_id'],
                'content': f"Loan {row['loan_id']}: ${row['loanamount']:,.2f} at {row['interestrate']:.2f}% for borrower with credit score {row['credit_score']}"
            })
    
    elif 'approved loan' in question_lower:
        approved = df[df['loan_status'] == 'Approved'].head(5)
        for _, row in approved.iterrows():
            relevant_docs.append({
                'loan_id': row['loan_id'],
                'content': f"Approved Loan {row['loan_id']}: ${row['loanamount']:,.2f} at {row['interestrate']:.2f}%"
            })
    
    return relevant_docs

def generate_answer_llm(question: str, context_docs: List[Dict[str, Any]]) -> str:
    """Generate LLM-style answer using context documents"""
    if not context_docs:
        return "No relevant loan documents found for your question."
    
    context_text = "\n".join([doc['content'] for doc in context_docs])
    
    # Simulate LLM response based on context
    if 'credit score' in question.lower():
        return f"""Based on the loan data analysis:

**Key Findings:**
- Found {len(context_docs)} relevant loan records
- These loans demonstrate strong creditworthiness patterns
- Higher credit scores typically correlate with better loan terms

**Sample Records:**
{context_text}

**Recommendation:** Borrowers with credit scores above 750 generally receive more favorable interest rates and approval odds."""
    
    else:
        return f"""**Analysis Results:**

Found {len(context_docs)} relevant loan records:

{context_text}

These records provide insight into the loan portfolio characteristics and can help with decision-making processes."""

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💰 Loan Query Assistant</h1>
        <p>Ask questions about loan data using Math Engine or LLM processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Mode selection
        mode = st.selectbox(
            "Select Processing Mode:",
            ["Math Engine", "LLM Engine"],
            help="Math Engine: Direct calculations\nLLM Engine: Contextual analysis"
        )
        
        st.markdown("---")
        
        # Data source
        data_source = st.radio(
            "Data Source:",
            ["Sample Data", "Upload CSV"],
            help="Use sample data or upload your own loan dataset"
        )
        
        # File upload
        uploaded_file = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload loan data CSV",
                type=['csv'],
                help="CSV should contain columns: loanamount, interestrate, credit_score, loan_status"
            )
    
    # Load data
    try:
        if data_source == "Upload CSV" and uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            st.success(f"✅ Loaded {len(df):,} records from uploaded file")
        else:
            df = load_sample_data()
            st.info(f"📊 Using sample dataset with {len(df):,} loan records")
        
        # Display data preview
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df.head(10))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Loans", f"{len(df):,}")
            with col2:
                avg_amount = df['loanamount'].mean()
                st.metric("Avg Loan Amount", f"${avg_amount:,.0f}")
            with col3:
                avg_rate = df["interestrate"].mean() if "interestrate" in df.columns else None
                st.metric("Avg Interest Rate", f"{avg_rate:.2f}%")
            with col4:
                approved_pct = (len(df[df['loan_status'] == 'Approved']) / len(df)) * 100
                st.metric("Approval Rate", f"{approved_pct:.1f}%")
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return
    
    # Query interface
    st.markdown("---")
    st.header("❓ Ask Your Question")
    
    # Sample questions
    st.markdown("**💡 Try these sample questions:**")
    sample_questions = [
        "What is the average loan amount?",
        "How many loans were approved?",
        "What is the highest interest rate?",
        "Show me the credit score distribution",
        "What is the lowest loan amount?"
    ]
    
    cols = st.columns(len(sample_questions))
    selected_sample = None
    for i, (col, question) in enumerate(zip(cols, sample_questions)):
        if col.button(f"📝 Q{i+1}", help=question):
            selected_sample = question
    
   # Make sure session_state.question exists
    if "question" not in st.session_state:
          st.session_state.question = ""

# If a sample question was clicked, override session state
    if selected_sample:
          st.session_state.question = selected_sample

# Show the input field with the current session question
    question = st.text_area(
        "Enter your question:",
        value=st.session_state.question,
        placeholder="e.g., What is the average loan amount for approved loans?",
        height=100,
        key="question_input"
    )

    
    # Process question
    if st.button("🔍 **Get Answer**", type="primary") and question.strip():
        with st.spinner("🔄 Processing your question..."):
            try:
                # ✅ CORRECTED CODE BLOCK
                if mode == "Math Engine":
                    response = generate_answer_smart(df, question)
                else:
                    response = generate_answer_smart(
                        df, question,
                        retrieve_similar_docs=retrieve_similar_docs,
                        generate_answer_llm=generate_answer_llm
                    )

                # ✅ Display result
                st.markdown("### 📤 Analysis Result")
                st.markdown(f"""
                <div class="answer-section">
                    <h4 style="color: #0369a1; margin: 0 0 1rem 0;">💬 Your Question:</h4>
                    <p style="color: #374151; font-style: italic; margin: 0 0 1.5rem 0;">"{question}"</p>
                    <h4 style="color: #0369a1; margin: 0 0 1rem 0;">🎯 Answer:</h4>
                    <div style="color: #1f2937; line-height: 1.6;">{response}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("❓ Ask Another Question"):
                    st.experimental_rerun()

            except Exception as e:
                st.error(f"❌ **Error processing your question:** {str(e)}")
                st.info("💡 **Tip:** Try rephrasing your question or check if your dataset has the required columns.")
                
                # Debug information
                with st.expander("🔧 Debug Information"):
                    st.text(traceback.format_exc())
    
    elif question.strip() == "":
        st.warning("⚠️ Please enter a question to get started!")

if __name__ == "__main__":
    main()
