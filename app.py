import streamlit as st
import os
import faiss
import numpy as np
from openai import OpenAI
from pdf_processing import extract_text_from_pdf
from vector_search import create_embeddings, search_similar_chunks
from sentence_transformers import SentenceTransformer

# Initialize LLM and embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
OPENAI_API_KEY = "your-openai-api-key"  # Replace with your OpenAI key
client = OpenAI(api_key=OPENAI_API_KEY)

# Streamlit UI
st.title("📄 ChatPDF: Interactive Document Assistant")
st.sidebar.subheader("Upload your PDF")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("PDF uploaded successfully!")
    
    # Extract text
    pdf_text = extract_text_from_pdf("temp.pdf")
    st.write("Extracted Text Preview:", pdf_text[:1000] + "...")
    
    # Create embeddings
    chunks, index = create_embeddings(pdf_text, embed_model)
    
    # User input for Q&A
    question = st.text_input("Ask a question about the document:")
    if question:
        matched_chunks = search_similar_chunks(question, embed_model, index, chunks)
        prompt = f"Context:\n{'\n'.join(matched_chunks)}\n\nUser Question: {question}\nAnswer:"
        response = client.completions.create(model="gpt-3.5-turbo", prompt=prompt, max_tokens=200)
        st.write("**Answer:**", response.choices[0].text.strip())

    # Summarization button
    if st.button("Summarize PDF"):
        prompt = f"Summarize the following document:\n{pdf_text[:5000]}\nSummary:"
        response = client.completions.create(model="gpt-3.5-turbo", prompt=prompt, max_tokens=250)
        st.write("**Summary:**", response.choices[0].text.strip())
