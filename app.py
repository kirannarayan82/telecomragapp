!pip install --upgrade transformers
import streamlit as st
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Initialize the retriever model
retriever = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the text generation model
generation_pipeline = pipeline('text2text-generation', model='facebook/bart-large-cnn')

st.title('RAG-based Customer Support Assistant')

query = st.text_input('Enter your query:')

# Load CSV files
queries_df = pd.read_csv('queries.csv')
responses_df = pd.read_csv('responses.csv')

documents = queries_df['query'].tolist()
responses = queries_df['response'].tolist()

def retrieve_docs(query, docs, top_k=2):
    embeddings = retriever.encode(docs, convert_to_tensor=True)
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    best_docs_idx = scores.topk(top_k)[1]
    best_docs = [docs[idx] for idx in best_docs_idx]
    best_responses = [responses[idx] for idx in best_docs_idx]
    return best_docs, best_responses

def generate_response(query, docs):
    input_text = f"Query: {query} Context: {' '.join(docs)}"
    result = generation_pipeline(input_text, max_length=200)
    return result[0]['generated_text']
print("this is the rag response")

if query:
    retrieved_docs, retrieved_resps = retrieve_docs(query, documents)
    response = generate_response(query, retrieved_docs)
    st.write(f"Response: {response}")
    for doc, resp in zip(retrieved_docs, retrieved_resps):
        st.write(f"Retrieved Query: {doc}\nResponse: {resp}")

else:
    print("Sorry unable to answer your query: will connect to a rep")

