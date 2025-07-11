# main.py - CLI to use rag_math_engine
import argparse
import pandas as pd
from rag_math_engine import generate_answer_smart
from llm_fallback import build_vector_store, retrieve_similar_docs, generate_answer_llm

parser = argparse.ArgumentParser(description="Ask smart math/LLM questions on tabular data")
parser.add_argument("--csv", required=True, help="Path to CSV dataset")
parser.add_argument("--mode", choices=["math", "llm"], default="math", help="Mode of answering")
parser.add_argument("--question", required=True, help="Your query")
args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.csv)

# LLM setup if needed
if args.mode == "llm":
    docs = [row.to_string() for _, row in df.iterrows()]
    build_vector_store(docs)

# Ask and print answer
response = generate_answer_smart(
    df,
    args.question,
    retrieve_similar_docs if args.mode == "llm" else None,
    generate_answer_llm if args.mode == "llm" else None
)
print("\nAnswer:\n", response)
