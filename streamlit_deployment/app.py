import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMultipleChoice

st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        background: -webkit-linear-gradient(45deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .subtitle {
        color: #A0AEC0;
        font-size: 1.2rem;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6C63FF 0%, #8A2BE2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(108, 99, 255, 0.3);
    }
    .prediction-card {
        background-color: #1A1A2E;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #6C63FF;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .rank-1 { border-left-color: #FFD700; }
    .rank-2 { border-left-color: #C0C0C0; }
    .rank-3 { border-left-color: #CD7F32; }
</style>
""", unsafe_allow_html=True)

def clean_text(t):
    if pd.isna(t) or not t: return ""
    t = str(t).lower()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def clean_prompt(p):
    p = re.sub(r'(?i)pick the best possible answer:\s*', '', str(p))
    p = re.sub(r'(?i)\s*(among the listed options|from the following choices|carefully)\.?\s*$', '', p)
    return clean_text(p)

@st.cache_resource(show_spinner=False)
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner=False)
def load_distilbert():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForMultipleChoice.from_pretrained('distilbert-base-uncased')
    model.eval()
    return tokenizer, model

def predict_sbert(prompt, options):
    model = load_sbert_model()
    prompt_clean = clean_prompt(prompt)
    options_clean = [clean_text(opt) for opt in options]
    
    prompt_emb = model.encode([prompt_clean], normalize_embeddings=True)
    option_embs = model.encode(options_clean, normalize_embeddings=True)
    
    similarities = np.dot(option_embs, prompt_emb.T).flatten()
    
    exp_sims = np.exp(similarities - np.max(similarities))
    probs = exp_sims / exp_sims.sum()
    
    top3_idx = np.argsort(-probs)[:3]
    return probs, top3_idx

def predict_distilbert(prompt, options):
    tokenizer, model = load_distilbert()
    prompt_clean = clean_prompt(prompt)
    
    encodings = []
    for opt in options:
        enc = tokenizer(prompt_clean, clean_text(opt), max_length=256, 
                       padding='max_length', truncation=True, return_tensors='pt')
        encodings.append(enc)
    
    input_ids = torch.stack([e['input_ids'].squeeze(0) for e in encodings]).unsqueeze(0)
    attention_mask = torch.stack([e['attention_mask'].squeeze(0) for e in encodings]).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    
    top3_idx = np.argsort(-probs)[:3]
    return probs, top3_idx

with st.sidebar:
    st.title("Smart MCQ Solver")
    st.markdown("AI-powered multiple choice question answering system.")
    
    st.markdown("### Model Selection")
    model_choice = st.selectbox(
        "Choose Inference Pipeline:",
        ["Sentence Transformer (Cosine Similarity)", "DistilBERT (Base Zero-Shot)"],
        index=0
    )
    
    st.divider()
    
    st.markdown("###Project Models Performance")
    st.markdown("The complete project involved 5 models. Evaluation on MAP@3:")
    
    perf_data = pd.DataFrame({
        "Model": ["DeBERTa-v3-Large", "DistilBERT (Tuned)", "MiniLM (Tuned)", "Sentence Transformer", "DistilBERT (Zero-Shot)"],
        "MAP@3": ["0.892", "0.841", "0.825", "0.640", "0.412"]
    })
    st.dataframe(perf_data, hide_index=True, use_container_width=True)
    
    st.info("**Deployment Note:** Due to Streamlit Community Cloud constraints (1GB RAM, No GPU), this app demonstrates methodology using lightweight pre-trained baseline models rather than the fine-tuned competitive weights.")
    
    st.divider()
    st.markdown("### About")
    st.markdown("Developed as part of a DL & GenAI Course Project (T2).")


st.markdown('<p class="main-title"> Smart MCQ Solver</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Evaluate prompts against choices utilizing lightweight transformer architecture.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### Enter Question & Options")
    with st.container(border=True):
        prompt = st.text_area("Question / Prompt", value="What is the powerhouse of the cell?", height=100)
        
        st.markdown("**Options:**")
        opt_a = st.text_input("A", value="Nucleus")
        opt_b = st.text_input("B", value="Mitochondria")
        opt_c = st.text_input("C", value="Ribosome")
        opt_d = st.text_input("D", value="Endoplasmic Reticulum")
        opt_e = st.text_input("E", value="Golgi Apparatus")
        
        options = [opt_a, opt_b, opt_c, opt_d, opt_e]
        labels = ["A", "B", "C", "D", "E"]
        
        solve_btn = st.button("Solve MCQ", use_container_width=True)

with col2:
    if solve_btn:
        if not prompt or not all(options):
            st.error("Please fill in the question and all 5 options.")
        else:
            with st.spinner(f"Running inference with {model_choice.split('(')[0].strip()}..."):
                start_time = time.time()
                
                if "Sentence Transformer" in model_choice:
                    probs, top3_idx = predict_sbert(prompt, options)
                else:
                    probs, top3_idx = predict_distilbert(prompt, options)
                    
                end_time = time.time()
                
            st.success(f"Inference completed in {end_time - start_time:.2f}s using {model_choice.split('(')[0].strip()}")
            
            st.markdown("### Top 3 Predictions")
            
            ranks = ["🥇", "🥈", "🥉"]
            css_classes = ["rank-1", "rank-2", "rank-3"]
            
            for i, idx in enumerate(top3_idx):
                prob = probs[idx]
                st.markdown(f"""
                <div class="prediction-card {css_classes[i]}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.1rem; font-weight: bold;">{ranks[i]} Option {labels[idx]}</span>
                        <span style="color: #6C63FF; font-weight: bold;">{prob*100:.1f}%</span>
                    </div>
                    <div style="margin-top: 0.5rem; color: #E2E8F0;">
                        {options[idx]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(prob))
            
            st.markdown("### Probability Distribution")
            chart_data = pd.DataFrame(
                {"Probability": probs}, 
                index=[f"{l}: {opt[:10]}{'...' if len(opt)>10 else ''}" for l, opt in zip(labels, options)]
            )
            st.bar_chart(chart_data, color="#6C63FF")
            
    else:
        st.markdown("### How it works")
        
        with st.expander("Sentence Transformer (Cosine Similarity)", expanded=True):
            st.markdown("""
            1. Cleans the prompt and all 5 options to remove special characters.
            2. Uses `all-MiniLM-L6-v2` to generate dense vector embeddings for the prompt and each option.
            3. Calculates Cosine Similarity between the prompt vector and option vectors.
            4. Applies softmax to normalize similarities into probabilities.
            5. Ranks and selects the top 3 options.
            """)
            
        with st.expander("DistilBERT (Base Zero-Shot)"):
            st.markdown("""
            1. Cleans the text inputs.
            2. Pairs the prompt with each option and tokenizes them using `distilbert-base-uncased`.
            3. Passes the pairs through the `AutoModelForMultipleChoice` architecture.
            4. The un-finetuned model outputs raw logits which are converted to probabilities via softmax.
            5. Ranks and selects the top 3 options based on raw zero-shot scoring.
            """)
