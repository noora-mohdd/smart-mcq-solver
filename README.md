# Smart MCQ Solver

Kaggle competition where we build systems that solve multiple-choice questions. Given a prompt and five options (A-E), the goal is to predict the top 3 most likely answers. Evaluation metric is MAP@3.

## Models

| Model | Approach | Val MAP@3 |
|-------|----------|-----------|
| **DistilBERT + RAG** | Fine-tuned distilbert-base with FAISS retrieval-augmented inference | **0.75** |
| **DeBERTa-v3 + LoRA** | LoRA fine-tuning on deberta-v3-base (r=16, alpha=32) | 0.73 |
| **LR + XGBoost Blend** | Handcrafted NLP features with 60/40 blend | 0.62 |
| **Bi-LSTM (from scratch)** | Custom tokenizer, word embeddings, bidirectional LSTM with attention | ~0.50 |
| **Sentence Transformer** | SBERT embeddings + PCA + logistic regression | 0.43 |

## Project Structure

    milestones/
        milestone-1.ipynb
        milestone-2.ipynb
        milestone-3.ipynb
        milestone-4.ipynb
        milestone-5.ipynb
    models/
        scratch-bilstm.ipynb
        sentence-transformer.ipynb
        lr-xgb-blend.ipynb
        deberta-lora.ipynb
        dl-23f2004192-notebook-t22026.ipynb
    report/
        report.pdf
    requirements.txt
    README.md

## How to Run

All notebooks are designed to run on Kaggle with GPU (P100/T4). Add the competition dataset (`smart-mcq-solver-challenge`) as input and set your WandB API key in Kaggle Secrets.

