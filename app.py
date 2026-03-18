import gradio as gr
import numpy as np
import os
from markitdown import MarkItDown
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

md = MarkItDown()
model = SentenceTransformer('all-MiniLM-L6-v2')
vector_db = {"chunks": [], "embeddings": []}

def process_pdf(file):
    if file is None: return "No file uploaded."
    try:
        result = md.convert(file.name)
        raw_text = result.text_content
        
        # Optimized Chunking for Financial Tables
        # Using 1000 characters with 200 overlap helps keep headers and numbers together
        chunks = [raw_text[i : i + 1000] for i in range(0, len(raw_text), 800)]
        
        embeddings = model.encode(chunks)
        vector_db["chunks"], vector_db["embeddings"] = chunks, embeddings
        return f"✅ Success! Processed {len(chunks)} chunks from the L'Oréal Report."
    except Exception as e: return f"❌ Error: {str(e)}"

def talk_to_pdf(message, history):
    if not vector_db["chunks"]: return "Please upload and index a PDF first!"
    
    query_vec = model.encode([message])
    embeddings = vector_db["embeddings"]
    
    # Cosine Similarity Calculation
    norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_vec)
    similarities = np.dot(embeddings, query_vec.T).flatten() / (norms * query_norm)
    
    best_idx = np.argmax(similarities)
    original_text = vector_db["chunks"][best_idx]
    
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(original_text)
        formatted_response = (
            f"### 🇺🇸 English Translation\n{translated_text}\n\n"
            f"--- \n"
            f"### 🇫🇷 Original French Source\n{original_text}"
        )
        return formatted_response
    except:
        return f"### 🇫🇷 Original Source (Translation Failed)\n\n{original_text}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📑 L'Oréal 2024 Financial Analyst Pro")
    gr.Markdown("### Multilingual RAG System: Retrieval-Augmented Generation")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload L'Oréal Report (PDF)", file_types=[".pdf"])
        process_btn = gr.Button("Index Report Content", variant="primary")
    
    status = gr.Textbox(label="System Status", interactive=False)
    chat = gr.ChatInterface(fn=talk_to_pdf)
    
    process_btn.click(fn=process_pdf, inputs=[pdf_input], outputs=[status])

if __name__ == "__main__":
    demo.launch()