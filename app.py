# app.py
import streamlit as st
import pandas as pd
import os
import time
from rag_math_engine import generate_answer_smart
from llm_fallback import retrieve_similar_docs, generate_answer_llm

# Load API key from secrets for Streamlit Cloud
if 'GOOGLE_API_KEY' in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ------------------------
# Custom CSS for Modern UI
# ------------------------
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Custom font for entire app */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e1e8ed;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed #cbd5e0;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea;
        background-color: #f8fafc;
    }
    
    /* Success/Info messages */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 8px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 8px;
    }
    
    /* Data table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Answer section styling */
    .answer-section {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #0ea5e9;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin-top: 1rem;
    }
    
    /* Progress bar */
    .stProgress .css-1cpxqw2 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stApp > header {height: 0;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .main-header p {
            font-size: 1rem;
        }
        .custom-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------
# Helper Functions
# ------------------------
def display_welcome_message():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Smart Loan Query Assistant</h1>
        <p>Advanced AI-powered loan analysis with intelligent data processing</p>
    </div>
    """, unsafe_allow_html=True)

def display_dataset_info(df):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">📊 Total Records</h3>
            <h2 style="color: #1f2937; margin: 0.5rem 0;">{}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #10b981; margin: 0;">📈 Columns</h3>
            <h2 style="color: #1f2937; margin: 0.5rem 0;">{}</h2>
        </div>
        """.format(len(df.columns)), unsafe_allow_html=True)
    
    with col3:
        numeric_cols = df.select_dtypes(include=['number']).columns
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #f59e0b; margin: 0;">🔢 Numeric</h3>
            <h2 style="color: #1f2937; margin: 0.5rem 0;">{}</h2>
        </div>
        """.format(len(numeric_cols)), unsafe_allow_html=True)
    
    with col4:
        categorical_cols = df.select_dtypes(include=['object']).columns
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ef4444; margin: 0;">📝 Categorical</h3>
            <h2 style="color: #1f2937; margin: 0.5rem 0;">{}</h2>
        </div>
        """.format(len(categorical_cols)), unsafe_allow_html=True)

def display_sample_questions():
    st.markdown("""
    <div class="custom-card">
        <h3 style="color: #374151; margin-bottom: 1rem;">💡 Sample Questions to Try</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
            <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <strong>📊 Statistical Analysis</strong><br>
                <em>"What is the approval rate for self-employed applicants?"</em>
            </div>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
                <strong>💰 Financial Insights</strong><br>
                <em>"Average loan amount by income level"</em>
            </div>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b;">
                <strong>🎯 Comparative Analysis</strong><br>
                <em>"Compare approval rates between male and female applicants"</em>
            </div>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #ef4444;">
                <strong>🔍 Detailed Breakdown</strong><br>
                <em>"Show distribution of loan purposes by approval status"</em>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------
# Main Application
# ------------------------
def main():
    # Load custom CSS
    load_custom_css()
    
    # Configure page
    st.set_page_config(
        page_title="Smart Loan Query Assistant",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Display welcome message
    display_welcome_message()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>⚙️ Configuration Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Mode selection with enhanced UI
        st.markdown("### 🤖 Processing Mode")
        mode = st.radio(
            "Choose your analysis approach:",
            ["Math Engine", "LLM Fallback"],
            help="Math Engine uses rule-based analysis, while LLM Fallback uses AI language models"
        )
        
        # Enhanced mode descriptions
        if mode == "Math Engine":
            st.success("🔢 **Math Engine Selected**\n\nFast, accurate rule-based analysis for numerical and statistical queries.")
        else:
            st.info("🧠 **LLM Fallback Selected**\n\nAI-powered natural language processing for complex queries.")
        
        st.markdown("---")
        
        # File uploader section
        st.markdown("### 📁 Dataset Upload")
        df_file = st.file_uploader(
            "Upload your CSV dataset",
            type=["csv"],
            help="Upload a CSV file containing loan data for analysis"
        )
        
        if df_file:
            file_size = len(df_file.getvalue())
            st.success(f"✅ **File uploaded successfully!**\n\nSize: {file_size/1024:.1f} KB")
        
        st.markdown("---")
        
        # Additional info
        st.markdown("""
        <div style="background: #f1f5f9; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <h4 style="color: #334155; margin: 0 0 0.5rem 0;">ℹ️ About This App</h4>
            <p style="color: #64748b; font-size: 0.9rem; margin: 0;">
                Advanced loan data analysis powered by AI. Upload your dataset and ask questions in natural language to get instant insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if df_file:
        try:
            # Load and display dataset
            with st.spinner("📊 Loading dataset..."):
                df = pd.read_csv(df_file)
                time.sleep(0.5)  # Small delay for better UX
            
            # Success message with fade-in effect
            st.balloons()
            st.success("🎉 Dataset loaded successfully! Ready for analysis.")
            
            # Dataset overview
            st.markdown("### 📋 Dataset Overview")
            display_dataset_info(df)
            
            # Data preview in expandable section
            with st.expander("📊 **View Data Preview**", expanded=False):
                st.markdown("**First 10 rows of your dataset:**")
                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Sample questions
            display_sample_questions()
            
            # Query section
            st.markdown("---")
            st.markdown("### 🔍 Ask Your Question")
            
            # Create two columns for better layout
            col1, col2 = st.columns([4, 1])
            
            with col1:
                question = st.text_input(
                    "Enter your question:",
                    placeholder="e.g., What is the approval rate for self-employed applicants?",
                    help="Ask any question about your loan dataset in natural language"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                analyze_button = st.button("🚀 Analyze", use_container_width=True)
            
            # Process question
            if question and (analyze_button or question):
                with st.spinner("🔄 Processing your question..."):
                    # Add progress bar for better UX
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Generate answer
                    try:
                        if mode == "Math Engine":
                            response = generate_answer_smart(df, question)
                        else:
                            response = generate_answer_smart(
                                df, question,
                                retrieve_similar_docs=retrieve_similar_docs,
                                generate_answer_llm=generate_answer_llm
                            )
                        
                        # Display answer with enhanced styling
                        st.markdown("### 📤 Analysis Result")
                        st.markdown(f"""
                        <div class="answer-section">
                            <h4 style="color: #0369a1; margin: 0 0 1rem 0;">💬 Your Question:</h4>
                            <p style="color: #374151; font-style: italic; margin: 0 0 1.5rem 0;">"{question}"</p>
                            <h4 style="color: #0369a1; margin: 0 0 1rem 0;">🎯 Answer:</h4>
                            <div style="color: #1f2937; line-height: 1.6;">{response}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add option to ask another question
                        if st.button("❓ Ask Another Question"):
                            st.experimental_rerun()
                    
                    except Exception as e:
                        st.error(f"❌ **Error processing your question:** {str(e)}")
                        st.info("💡 **Tip:** Try rephrasing your question or check if your dataset has the required columns.")
            
            elif question:
                st.info("👆 Click the 'Analyze' button to process your question!")
    
    else:
        # Welcome screen when no file is uploaded
        st.markdown("""
        <div class="custom-card">
            <div style="text-align: center; padding: 2rem;">
                <h2 style="color: #374151; margin-bottom: 1rem;">🚀 Welcome to Smart Loan Analysis</h2>
                <p style="color: #6b7280; font-size: 1.1rem; margin-bottom: 2rem;">
                    Upload your CSV dataset to start analyzing loan data with AI-powered insights
                </p>
                <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                    <div style="background: #f0f9ff; padding: 1.5rem; border-radius: 12px; min-width: 200px;">
                        <h3 style="color: #0369a1; margin: 0 0 0.5rem 0;">📊 Smart Analysis</h3>
                        <p style="color: #374151; margin: 0; font-size: 0.9rem;">AI-powered data processing</p>
                    </div>
                    <div style="background: #f0fdf4; padding: 1.5rem; border-radius: 12px; min-width: 200px;">
                        <h3 style="color: #16a34a; margin: 0 0 0.5rem 0;">🎯 Natural Language</h3>
                        <p style="color: #374151; margin: 0; font-size: 0.9rem;">Ask questions in plain English</p>
                    </div>
                    <div style="background: #fef3c7; padding: 1.5rem; border-radius: 12px; min-width: 200px;">
                        <h3 style="color: #d97706; margin: 0 0 0.5rem 0;">⚡ Instant Results</h3>
                        <p style="color: #374151; margin: 0; font-size: 0.9rem;">Get answers in seconds</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: #374151; margin-bottom: 1rem;">📝 How to Get Started</h3>
            <ol style="color: #6b7280; line-height: 1.8;">
                <li><strong>Upload your CSV file</strong> using the sidebar file uploader</li>
                <li><strong>Choose your processing mode</strong> (Math Engine or LLM Fallback)</li>
                <li><strong>Ask questions</strong> about your loan data in natural language</li>
                <li><strong>Get instant insights</strong> with detailed analysis and visualizations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
