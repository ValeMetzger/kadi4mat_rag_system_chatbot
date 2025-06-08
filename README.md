---
title: Kadi4mat Rag System
emoji: 🆒
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10.16
app_file: start.py
pinned: false
---

# KADI4Mat RAG Chatbot 🤖📚

A Retrieval-Augmented Generation (RAG) chatbot demo built using Hugging Face Transformers and Gradio. This project is part of the **KADI4Mat initiative**, designed to explore the integration of domain-specific knowledge into large language model applications.

## 🚀 Demo

Try the chatbot live on [Hugging Face Spaces](https://huggingface.co/spaces/ValeMetzger/kadi4mat_rag_system_chatbot)

## 🧠 Features

- 🔎 **Retrieval-Augmented Generation** using custom document embeddings
- 💬 **Conversational interface** powered by Gradio
- 🧾 Uses **FAISS vector search** for semantic document matching
- 🏗️ Modular and extensible Python structure
- 📄 PDF/Text file ingestion and processing
- 🇬🇧 English and German language support (depending on model used)

## 🔧 Tech Stack

- Python
- Hugging Face Transformers & Datasets
- Gradio
- FAISS
- SentenceTransformers
- PyMuPDF

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/ValeMetzger/kadi4mat_rag_system_chatbot.git
cd kadi4mat_rag_system_chatbot