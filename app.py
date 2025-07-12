import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import traceback
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# Configuration & Constants
# ------------------------
DEFAULT_INTEREST_RATE = 0.07  # 7% annual rate
SAMPLE_DATA_SIZE = 1000

# ------------------------
# Streamlit App Config
# ------------------------
st.set_page_config(
    page_title="CreditWise RAG Chatbot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------
# Custom CSS Styling
# ------------------------
def load_custom_css():
    st.markdown("""
        <style>
        .block-container {
            padding: 2rem 4rem;
        }
        .stApp > header {
            padding-top: 2rem;
        }
        .answer-section {
            background-color: #f1f5f9;
            border-left: 5px solid #0ea5e9;
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            margin-top: 1.5rem;
        }
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        .main-header {
            background: linear-gradient(90deg, #0369a1 0%, #0284c7 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .error-box {
            background: #fee2e2;
            border: 1px solid #fca5a5;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #dc2626;
        }
        .success-box {
            background: #d1fae5;
            border: 1px solid #86efac;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #059669;
        }
        .warning-box {
            background: #fef3c7;
            border: 1px solid #fcd34d;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #d97706;
        }
        .quick-question-btn {
            background: #f0f9ff;
            border: 1px solid #0ea5e9;
            border-radius: 6px;
            padding: 0.5rem;
            margin: 0.2rem;
            text-align: center;
            font-size: 0.85rem;
            transition: all 0.3s ease;
        }
        .quick-question-btn:hover {
            background: #0ea5e9;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# ------------------------
# Data Generation Functions
# ------------------------
@st.cache_data
def generate_sample_data(size: int = SAMPLE_DATA_SIZE) -> pd.DataFrame:
    """Generate comprehensive sample loan data"""
    try:
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic loan data
        data = {
            'loan_id': [f'LN{str(i).zfill(6)}' for i in range(1, size + 1)],
            'borrower_name': [f'Borrower_{i}' for i in range(1, size + 1)],
            'loanamount': np.random.lognormal(mean=10.8, sigma=0.5, size=size).round(2),
            'interestrate': np.random.normal(7.5, 2.5, size).clip(3, 15).round(2),
            'term_months': np.random.choice([12, 24, 36, 48, 60], size),
            'credit_score': np.random.randint(300, 850, size),
            'annual_income': np.random.lognormal(mean=11.0, sigma=0.4, size=size).round(2),
            'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], size, p=[0.75, 0.2, 0.05]),
            'loan_status': np.random.choice(['Approved', 'Pending', 'Rejected'], size, p=[0.65, 0.15, 0.2]),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size, p=[0.3, 0.4, 0.25, 0.05]),
            'debt_to_income_ratio': np.random.uniform(0.1, 0.6, size).round(3),
            'loan_purpose': np.random.choice(['Home Purchase', 'Car Loan', 'Business', 'Education', 'Personal'], size, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        }
        
        df = pd.DataFrame(data)
        
        # Calculate monthly payment using loan formula
        principal = df['loanamount']
        rate = df['interestrate'] / 100 / 12
        months = df['term_months']
        
        # Handle edge cases where rate is 0
        mask = rate > 0
        df['monthly_payment'] = 0.0
        df.loc[mask, 'monthly_payment'] = (
            principal[mask] * rate[mask] * (1 + rate[mask])**months[mask]
        ) / ((1 + rate[mask])**months[mask] - 1)
        df.loc[~mask, 'monthly_payment'] = principal[~mask] / months[~mask]
        
        df['monthly_payment'] = df['monthly_payment'].round(2)
        
        # Add some realistic correlations
        # Higher credit scores tend to get better rates
        high_credit_mask = df['credit_score'] >= 750
        df.loc[high_credit_mask, 'interestrate'] = df.loc[high_credit_mask, 'interestrate'] * 0.85
        
        # Higher income correlates with higher loan amounts
        high_income_mask = df['annual_income'] >= 80000
        df.loc[high_income_mask, 'loanamount'] = df.loc[high_income_mask, 'loanamount'] * 1.2
        
        return df
        
    except Exception as e:
        st.error(f"Error generating sample data: {str(e)}")
        # Return minimal fallback data
        return pd.DataFrame({
            'loan_id': ['LN000001'],
            'loanamount': [50000],
            'interestrate': [7.5],
            'credit_score': [700],
            'loan_status': ['Approved']
        })

# ------------------------
# Analysis Engine Functions
# ------------------------
def safe_column_access(df: pd.DataFrame, column_variants: List[str]) -> Optional[str]:
    """Safely access columns with different naming conventions"""
    for variant in column_variants:
        if variant in df.columns:
            return variant
    return None

def generate_answer_smart(df: pd.DataFrame, question: str, **kwargs) -> str:
    """Smart answer generation with robust error handling"""
    try:
        question_lower = question.lower()
        
        # Column mapping for different naming conventions
        amount_col = safe_column_access(df, ['loanamount', 'loan_amount', 'amount'])
        rate_col = safe_column_access(df, ['interestrate', 'interest_rate', 'rate'])
        credit_col = safe_column_access(df, ['credit_score', 'creditscore', 'score'])
        status_col = safe_column_access(df, ['loan_status', 'loanstatus', 'status'])
        income_col = safe_column_access(df, ['annual_income', 'income', 'salary'])
        
        # Statistical queries
        if any(word in question_lower for word in ['average', 'mean', 'avg']):
            if 'loan' in question_lower and 'amount' in question_lower and amount_col:
                avg_amount = df[amount_col].mean()
                return f"The average loan amount is **${avg_amount:,.2f}**"
            elif 'interest' in question_lower and rate_col:
                avg_rate = df[rate_col].mean()
                return f"The average interest rate is **{avg_rate:.2f}%**"
            elif 'credit' in question_lower and credit_col:
                avg_score = df[credit_col].mean()
                return f"The average credit score is **{avg_score:.0f}**"
            elif 'income' in question_lower and income_col:
                avg_income = df[income_col].mean()
                return f"The average annual income is **${avg_income:,.2f}**"
        
        # Count queries
        elif any(word in question_lower for word in ['how many', 'count', 'number of']):
            if 'approved' in question_lower and status_col:
                approved_count = len(df[df[status_col] == 'Approved'])
                total_count = len(df)
                percentage = (approved_count / total_count) * 100
                return f"There are **{approved_count:,}** approved loans out of {total_count:,} total loans (**{percentage:.1f}%**)"
            elif 'rejected' in question_lower and status_col:
                rejected_count = len(df[df[status_col] == 'Rejected'])
                total_count = len(df)
                percentage = (rejected_count / total_count) * 100
                return f"There are **{rejected_count:,}** rejected loans out of {total_count:,} total loans (**{percentage:.1f}%**)"
        
        # Highest/lowest queries
        elif any(word in question_lower for word in ['highest', 'maximum', 'max']):
            if 'loan' in question_lower and amount_col:
                max_amount = df[amount_col].max()
                return f"The highest loan amount is **${max_amount:,.2f}**"
            elif 'interest' in question_lower and rate_col:
                max_rate = df[rate_col].max()
                return f"The highest interest rate is **{max_rate:.2f}%**"
            elif 'credit' in question_lower and credit_col:
                max_score = df[credit_col].max()
                return f"The highest credit score is **{max_score}**"
        
        elif any(word in question_lower for word in ['lowest', 'minimum', 'min']):
            if 'loan' in question_lower and amount_col:
                min_amount = df[amount_col].min()
                return f"The lowest loan amount is **${min_amount:,.2f}**"
            elif 'interest' in question_lower and rate_col:
                min_rate = df[rate_col].min()
                return f"The lowest interest rate is **{min_rate:.2f}%**"
            elif 'credit' in question_lower and credit_col:
                min_score = df[credit_col].min()
                return f"The lowest credit score is **{min_score}**"
        
        # Distribution analysis
        elif 'distribution' in question_lower or 'breakdown' in question_lower:
            if 'credit' in question_lower and credit_col:
                excellent = len(df[df[credit_col] >= 750])
                good = len(df[(df[credit_col] >= 700) & (df[credit_col] < 750)])
                fair = len(df[(df[credit_col] >= 650) & (df[credit_col] < 700)])
                poor = len(df[df[credit_col] < 650])
                
                return f"""**Credit Score Distribution:**
- Excellent (750+): **{excellent:,}** borrowers ({excellent/len(df)*100:.1f}%)
- Good (700-749): **{good:,}** borrowers ({good/len(df)*100:.1f}%)
- Fair (650-699): **{fair:,}** borrowers ({fair/len(df)*100:.1f}%)
- Poor (<650): **{poor:,}** borrowers ({poor/len(df)*100:.1f}%)"""
            
            elif 'status' in question_lower and status_col:
                status_counts = df[status_col].value_counts()
                breakdown = []
                for status, count in status_counts.items():
                    percentage = (count / len(df)) * 100
                    breakdown.append(f"- {status}: **{count:,}** loans ({percentage:.1f}%)")
                return f"**Loan Status Distribution:**\n" + "\n".join(breakdown)
        
        # Education-based queries
        elif 'education' in question_lower and 'education_level' in df.columns:
            if 'graduate' in question_lower or 'bachelor' in question_lower or 'master' in question_lower or 'phd' in question_lower:
                higher_ed = df[df['education_level'].isin(['Bachelor', 'Master', 'PhD'])]
                if amount_col:
                    avg_amount = higher_ed[amount_col].mean()
                    return f"The average loan amount for graduates is **${avg_amount:,.2f}** (based on {len(higher_ed):,} records)"
        
        # Comparison queries
        elif 'vs' in question_lower or 'versus' in question_lower or 'compare' in question_lower:
            if 'approved' in question_lower and 'rejected' in question_lower and status_col and amount_col:
                approved_avg = df[df[status_col] == 'Approved'][amount_col].mean()
                rejected_avg = df[df[status_col] == 'Rejected'][amount_col].mean()
                return f"""**Loan Amount Comparison:**
- Approved loans: **${approved_avg:,.2f}** average
- Rejected loans: **${rejected_avg:,.2f}** average
- Difference: **${abs(approved_avg - rejected_avg):,.2f}**"""
        
        # Fallback with dataset summary
        available_cols = [col for col in [amount_col, rate_col, credit_col, status_col] if col]
        return f"""I found **{len(df):,}** loan records in the dataset.

**Available data includes:**
{chr(10).join([f'• {col.replace("_", " ").title()}' for col in available_cols])}

**Try asking questions like:**
• What is the average loan amount?
• How many loans were approved?
• What is the credit score distribution?
• Compare approved vs rejected loans"""
    
    except Exception as e:
        return f"Error processing your question: {str(e)}. Please try rephrasing your question."

def retrieve_similar_docs(df: pd.DataFrame, question: str) -> List[Dict[str, Any]]:
    """Retrieve relevant loan records for LLM processing"""
    try:
        question_lower = question.lower()
        relevant_docs = []
        
        # Column mapping
        amount_col = safe_column_access(df, ['loanamount', 'loan_amount', 'amount'])
        rate_col = safe_column_access(df, ['interestrate', 'interest_rate', 'rate'])
        credit_col = safe_column_access(df, ['credit_score', 'creditscore', 'score'])
        status_col = safe_column_access(df, ['loan_status', 'loanstatus', 'status'])
        
        # Filter based on question context
        if 'high credit' in question_lower and credit_col:
            high_credit = df[df[credit_col] >= 750].head(5)
            for _, row in high_credit.iterrows():
                relevant_docs.append({
                    'loan_id': row.get('loan_id', 'N/A'),
                    'content': f"High credit score loan: ${row.get(amount_col, 0):,.2f} at {row.get(rate_col, 0):.2f}% for credit score {row.get(credit_col, 0)}"
                })
        
        elif 'approved' in question_lower and status_col:
            approved = df[df[status_col] == 'Approved'].head(5)
            for _, row in approved.iterrows():
                relevant_docs.append({
                    'loan_id': row.get('loan_id', 'N/A'),
                    'content': f"Approved loan: ${row.get(amount_col, 0):,.2f} at {row.get(rate_col, 0):.2f}%"
                })
        
        elif 'large' in question_lower or 'big' in question_lower and amount_col:
            large_loans = df.nlargest(5, amount_col)
            for _, row in large_loans.iterrows():
                relevant_docs.append({
                    'loan_id': row.get('loan_id', 'N/A'),
                    'content': f"Large loan: ${row.get(amount_col, 0):,.2f} at {row.get(rate_col, 0):.2f}%"
                })
        
        return relevant_docs
        
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

def generate_answer_llm(question: str, context_docs: List[Dict[str, Any]]) -> str:
    """Generate LLM-style contextual answer"""
    try:
        if not context_docs:
            return "No relevant loan documents found for your question."
        
        context_text = "\n".join([f"• {doc['content']}" for doc in context_docs])
        
        # Enhanced contextual responses
        if 'credit score' in question.lower():
            return f"""**Credit Score Analysis:**

Based on the loan data analysis, here are the key findings:

**Sample High Credit Score Loans:**
{context_text}

**Insights:**
• Borrowers with credit scores above 750 typically receive better loan terms
• Higher credit scores correlate with lower interest rates
• Approval rates are significantly higher for excellent credit scores

**Recommendation:** Maintaining a credit score above 750 can result in substantial savings over the loan term."""
        
        elif 'approved' in question.lower():
            return f"""**Approved Loan Analysis:**

**Sample Approved Loans:**
{context_text}

**Key Patterns:**
• Approved loans show consistent payment capacity
• Interest rates vary based on creditworthiness
• Loan amounts are typically aligned with income levels

**Insight:** Approved loans demonstrate strong correlation between borrower profile and loan terms."""
        
        elif 'large' in question.lower() or 'big' in question.lower():
            return f"""**Large Loan Analysis:**

**Largest Loans in Portfolio:**
{context_text}

**Characteristics:**
• Large loans typically require higher credit scores
• Interest rates may be competitive due to loan size
• These loans often represent significant financial commitments

**Note:** Large loans require thorough risk assessment and documentation."""
        
        else:
            return f"""**Loan Portfolio Analysis:**

**Relevant Loan Records:**
{context_text}

**Summary:**
• Found {len(context_docs)} relevant loan records
• Data shows diverse loan characteristics
• Analysis can help inform lending decisions

**Recommendation:** Consider the context of these loans when making policy decisions."""
            
    except Exception as e:
        return f"Error generating contextual answer: {str(e)}"

# ------------------------
# UI Helper Functions
# ------------------------
def display_dataset_info(df: pd.DataFrame):
    """Display comprehensive dataset information"""
    try:
        st.subheader("📊 Dataset Overview")
        
        # Display data preview
        st.write("**Sample Data:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            st.metric("Total Columns", f"{df.shape[1]}")
        
        with col3:
            # Find loan amount column
            amount_col = safe_column_access(df, ['loanamount', 'loan_amount', 'amount'])
            if amount_col:
                avg_amount = df[amount_col].mean()
                st.metric("Avg Loan Amount", f"${avg_amount:,.0f}")
            else:
                st.metric("Avg Loan Amount", "N/A")
        
        with col4:
            # Find interest rate column
            rate_col = safe_column_access(df, ['interestrate', 'interest_rate', 'rate'])
            if rate_col:
                avg_rate = df[rate_col].mean()
                st.metric("Avg Interest Rate", f"{avg_rate:.2f}%")
            else:
                st.metric("Avg Interest Rate", "N/A")
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            credit_col = safe_column_access(df, ['credit_score', 'creditscore', 'score'])
            if credit_col:
                avg_credit = df[credit_col].mean()
                st.metric("Avg Credit Score", f"{avg_credit:.0f}")
            else:
                st.metric("Avg Credit Score", "N/A")
        
        with col6:
            status_col = safe_column_access(df, ['loan_status', 'loanstatus', 'status'])
            if status_col:
                approved_pct = (len(df[df[status_col] == 'Approved']) / len(df)) * 100
                st.metric("Approval Rate", f"{approved_pct:.1f}%")
            else:
                st.metric("Approval Rate", "N/A")
        
        with col7:
            if 'monthly_payment' in df.columns:
                avg_payment = df['monthly_payment'].mean()
                st.metric("Avg Monthly Payment", f"${avg_payment:,.0f}")
            else:
                st.metric("Avg Monthly Payment", "N/A")
        
        with col8:
            if 'annual_income' in df.columns:
                avg_income = df['annual_income'].mean()
                st.metric("Avg Annual Income", f"${avg_income:,.0f}")
            else:
                st.metric("Avg Annual Income", "N/A")
        
        st.markdown("---")
        
    except Exception as e:
        st.error(f"Error displaying dataset info: {str(e)}")

def load_and_process_data(file_path: Optional[str] = None, uploaded_file = None) -> tuple:
    """Load and process data with comprehensive error handling"""
    try:
        df = None
        
        if uploaded_file is not None:
            # Handle uploaded file
            df = pd.read_csv(uploaded_file)
            source = "uploaded file"
            st.success(f"✅ Successfully loaded {len(df):,} records from uploaded file")
            
        elif file_path and os.path.exists(file_path):
            # Handle file path
            df = pd.read_csv(file_path)
            source = "sample dataset file"
            st.success(f"✅ Successfully loaded {len(df):,} records from {source}")
            
        else:
            # Generate sample data as fallback
            df = generate_sample_data()
            source = "generated sample data"
            st.info(f"📊 Using {source} with {len(df):,} records")
        
        if df is None or df.empty:
            raise ValueError("No data loaded")
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Calculate interest rate safely
        rate_col = safe_column_access(df, ['interestrate', 'interest_rate', 'rate'])
        if rate_col:
            # Convert percentage to decimal monthly rate
            rate = df[rate_col] / 100 / 12
        else:
            st.warning(f"⚠️ Interest rate column not found in {source}. Using default {DEFAULT_INTEREST_RATE*100:.0f}%.")
            rate = pd.Series([DEFAULT_INTEREST_RATE / 12] * len(df))
        
        return df, rate
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        # Return sample data as last resort
        try:
            df = generate_sample_data(100)  # Smaller sample for error cases
            rate = pd.Series([DEFAULT_INTEREST_RATE / 12] * len(df))
            st.warning("🔄 Loaded minimal sample data as fallback")
            return df, rate
        except:
            st.error("💥 Critical error: Unable to load any data")
            return None, None

def validate_dataset(df: pd.DataFrame) -> bool:
    """Validate dataset and provide feedback"""
    try:
        if df is None or df.empty:
            st.error("❌ Dataset is empty")
            return False
        
        # Check for basic loan data columns
        amount_col = safe_column_access(df, ['loanamount', 'loan_amount', 'amount'])
        if not amount_col:
            st.warning("⚠️ No loan amount column found. Analysis may be limited.")
        
        # Check for non-numeric data in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            st.warning("⚠️ No numeric columns found. Limited analysis available.")
        
        # Check for missing values
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 50:
            st.warning(f"⚠️ Dataset has {missing_pct:.1f}% missing values")
        elif missing_pct > 0:
            st.info(f"ℹ️ Dataset has {missing_pct:.1f}% missing values")
        
        return True
        
    except Exception as e:
        st.error(f"Error validating dataset: {str(e)}")
        return False

# ------------------------
# Main Application
# ------------------------
def main():
    """Main application function"""
    try:
        # Load custom CSS
        load_custom_css()
        
        # Main header
        st.markdown("""
        <div class="main-header">
            <h1>🤖 CreditWise: RAG Chatbot for Loan Data</h1>
            <p>Intelligent loan data analysis using Math Engine and LLM processing</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar configuration
        st.sidebar.header("⚙️ Configuration")
        
        # Processing mode selection
        processing_mode = st.sidebar.selectbox(
            "🔧 Processing Mode:",
            ["Math Engine", "LLM Engine"],
            help="Math Engine: Direct statistical analysis\nLLM Engine: Contextual analysis with document retrieval"
        )
        
        st.sidebar.markdown("---")
        
        # Data source selection
        data_source = st.sidebar.radio(
            "📂 Data Source:",
            ["Generate Sample Data", "Upload CSV File", "Load from File Path"],
            help="Choose how to load your loan data"
        )
        
        # File upload interface
        uploaded_file = None
        file_path = None
        
        if data_source == "Upload CSV File":
            uploaded_file = st.sidebar.file_uploader(
                "📄 Upload CSV File",
                type=['csv'],
                help="Upload a CSV file containing loan data"
            )
            
            if uploaded_file:
                st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
        
        elif data_source == "Load from File Path":
            file_path = st.sidebar.text_input(
                "📁 File Path:",
                value="dataset/Training Dataset.csv",
                help="Enter the path to your CSV file"
            )
            
            if file_path and os.path.exists(file_path):
                st.sidebar.success(f"✅ File found: {file_path}")
            elif file_path:
                st.sidebar.warning(f"⚠️ File not found: {file_path}")
        
        # Advanced settings
        with st.sidebar.expander("🔬 Advanced Settings"):
            show_debug = st.checkbox("Show Debug Information", value=False)
            max_results = st.slider("Max Results to Display", 1, 20, 5)
            
        st.sidebar.markdown("---")
        st.sidebar.markdown("**💡 Sample Questions:**")
        st.sidebar.markdown("""
        • What is the average loan amount?
        • How many loans were approved?
        • Show credit score distribution
        • Compare approved vs rejected loans
        • What is the highest interest rate?
        """)
        
        # Load and process data
        df, rate = load_and_process_data(file_path, uploaded_file)
        
        if df is not None and validate_dataset(df):
            # Display dataset information
            display_dataset_info(df)
            
            # Question input interface
            st.header("💬 Ask Your Question")

            # Initialize session state for selected question
            if 'selected_question' not in st.session_state:
                st.session_state.selected_question = ""

            # Sample question buttons
            st.markdown("**🚀 Quick Start Questions:**")

            sample_questions = [
                "What is the average loan amount?",
                "How many loans were approved?",
                "Show me the credit score distribution",
                "What is the highest interest rate?",
                "Compare approved vs rejected loans"
            ]

            cols = st.columns(len(sample_questions))

            for i, (col, question_text) in enumerate(zip(cols, sample_questions)):
                if col.button(f"💡 Q{i+1}", help=question_text, key=f"sample_q_{i}"):
                    st.session_state.selected_question = question_text
                    st.rerun()

            # Question input
            question = st.text_area(
                "✍️ Enter your question:",
                value=st.session_state.selected_question,
                placeholder="e.g., What is the average loan amount for approved loans with credit scores above 750?",
                height=100,
                key="question_input"
            )

            # Sync manual typing with session state
            if question != st.session_state.selected_question:
                st.session_state.selected_question = question

            # Analysis button
            col1, col2 = st.columns([3, 1])
            with col1:
                analyze_button = st.button("🔍 **Analyze Question**", type="primary")
            with col2:
                if st.button("🔄 Clear"):
                    st.session_state.selected_question = ""
                    st.rerun()
            
            # Process question
            if question.strip() and analyze_button:
                with st.spinner("🔄 Processing your question..."):
                    # Progress bar for visual feedback
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    try:
                        # Generate response based on processing mode
                        if processing_mode == "Math Engine":
                            response = generate_answer_smart(df, question)
                        else:  # LLM Engine
                            # Retrieve relevant documents
                            similar_docs = retrieve_similar_docs(df, question)
                            # Generate LLM response
                            llm_response = generate_answer_llm(question, similar_docs)
                            # Combine with mathematical analysis
                            math_response = generate_answer_smart(df, question)
                            response = f"{llm_response}\n\n---\n\n**📊 Additional Statistical Analysis:**\n{math_response}"
                        
                        # Display results
                        st.markdown("### 📤 Analysis Results")
                        st.markdown(f"""
                        <div class="answer-section">
                            <h4 style="color: #0369a1; margin: 0 0 1rem 0;">💬 Your Question:</h4>
                            <p style="color: #374151; font-style: italic; margin: 0 0 1.5rem 0; font-size: 16px;">"{question}"</p>
                            <h4 style="color: #0369a1; margin: 0 0 1rem 0;">🎯 Answer ({processing_mode}):</h4>
                            <div style="color: #1f2937; line-height: 1.6; font-size: 15px;">{response}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Action buttons
                        st.markdown("### 🎯 Next Steps")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("❓ Ask Another Question", key="ask_another"):
                                st.session_state.selected_question = ""
                                st.rerun()
                        
                        with col2:
                            if st.button("📊 View Dataset Again", key="view_dataset"):
                                st.rerun()
                        
                        with col3:
                            if st.button("🔄 Switch Mode", key="switch_mode"):
                                st.rerun()
                        
                        # Debug information
                        if show_debug:
                            with st.expander("🔧 Debug Information"):
                                st.write("**Question Processing:**")
                                st.write(f"- Original question: {question}")
                                st.write(f"- Processing mode: {processing_mode}")
                                st.write(f"- Dataset shape: {df.shape}")
                                st.write(f"- Available columns: {list(df.columns)}")
                                
                                if processing_mode == "LLM Engine":
                                    st.write(f"- Retrieved documents: {len(similar_docs)}")
                                    if similar_docs:
                                        st.write("- Sample documents:")
                                        for i, doc in enumerate(similar_docs[:3]):
                                            st.write(f"  {i+1}. {doc['content']}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing your question: {str(e)}")
                        st.info("💡 **Troubleshooting Tips:**")
                        st.info("• Try rephrasing your question")
                        st.info("• Check if the required data columns exist")
                        st.info("• Ensure your dataset has numeric data for calculations")
                        
                        if show_debug:
                            with st.expander("🔧 Technical Details"):
                                st.text(traceback.format_exc())
            
            elif question.strip() == "":
                st.warning("⚠️ Please enter a question to analyze the data!")
                
        else:
            st.error("💥 Unable to load or process data. Please check your data source.")
            
    except Exception as e:
        st.error(f"💥 Critical application error: {str(e)}")
        st.info("🔄 Please refresh the page to restart the application.")
        
        if st.button("🔄 Restart Application"):
            st.rerun()

# ------------------------
# Application Entry Point
# ------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"💥 Failed to start application: {str(e)}")
        st.markdown("""
        ### 🚨 Application Startup Error
        
        The application encountered an error during startup. This might be due to:
        - Missing dependencies
        - Invalid configuration
        - System resources issues
        
        **💡 Quick Fix:**
        1. Ensure you have Streamlit installed: `pip install streamlit pandas numpy`
        2. Run the app with: `streamlit run app.py`
        3. Check your Python environment
        
        **🔄 If the problem persists, try restarting your Python environment.**
        """)
