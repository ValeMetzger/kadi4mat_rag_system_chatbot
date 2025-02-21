"""
This is a demo to show how to use OAuth2 to connect an application to Kadi.

Read Section "OAuth2 Tokens" in Kadi documents.
Ref: https://kadi.readthedocs.io/en/stable/httpapi/intro.html#oauth2-tokens

Notes:
1. register an application in Kadi (Setting->Applications)
    - Name: KadiOAuthTest
    - Website URL: http://127.0.0.1:7860
    - Redirect URIs: http://localhost:7860/auth
    
And you will get Client ID and Client Secret, note them down and set in this file.

2. Start this app, and open browser with address "http://localhost:7860/"
  - if you are starting this app on Huggingface, use "start.py" instead.
"""

import json
import uvicorn
import gradio as gr
import kadi_apy
import pymupdf
import numpy as np
import faiss
import os
import tempfile
import pymupdf
from fastapi import FastAPI, Depends
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import Request
from kadi_apy import KadiManager
from requests.compat import urljoin
from typing import List, Tuple, Optional, Dict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import mimetypes
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document as DocxReader
import pandas as pd
import chromadb
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
from typing import List, Tuple
import re
import unicodedata
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from functools import wraps
import time
import random
import asyncio

# Kadi OAuth settings
load_dotenv()
KADI_CLIENT_ID = os.environ["KADI_CLIENT_ID"]
KADI_CLIENT_SECRET = os.environ["KADI_CLIENT_SECRET"]
SECRET_KEY = os.environ["SECRET_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
huggingfacehub_api_token = os.environ["huggingfacehub_api_token"]

from huggingface_hub import login

login(token=huggingfacehub_api_token)

# Set up OAuth
app = FastAPI()
oauth = OAuth()

# Set Kadi instance
instance = "my_instance"  # "demo kit instance"
host = "https://demo-kadi4mat.iam.kit.edu"

# Register oauth
base_url = host
oauth.register(
    name="kadi4mat",
    client_id=KADI_CLIENT_ID,
    client_secret=KADI_CLIENT_SECRET,
    api_base_url=f"{base_url}/api",
    access_token_url=f"{base_url}/oauth/token",
    authorize_url=f"{base_url}/oauth/authorize",
    access_token_params={
        "client_id": KADI_CLIENT_ID,
        "client_secret": KADI_CLIENT_SECRET,
    },
)

# Global LLM client
client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=huggingfacehub_api_token
)

# Mixed-usage of huggingface client and local model for showing 2 possibilities
embeddings_client = InferenceClient(
    model="sentence-transformers/all-mpnet-base-v2", token=huggingfacehub_api_token
)
embeddings_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2", trust_remote_code=True
)


class DocumentProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    def preprocess_document(self, text: str) -> str:
        # Remove technical markers and artifacts
        text = re.sub(r'={2,}.*?={2,}', '', text)  # Remove === sections
        text = re.sub(r'\[.*?\]', '', text)  # Remove [] sections
        text = re.sub(r'`{1,}', '', text)     # Remove backticks
        text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
        text = text.strip()
        return text
    
    def chunk_document(self, text: str, max_chunk_size: int = 512, min_chunk_size: int = 100) -> List[dict]:
        """Improved chunking without NLTK dependency."""
        chunks = []
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Get paragraph tokens
            tokens = self.tokenizer.encode(para)
            para_length = len(tokens)
            
            # If paragraph is too long, split by sentences using regex
            if para_length > max_chunk_size:
                # Simple sentence splitting using regex
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                        
                    sent_tokens = self.tokenizer.encode(sent)
                    sent_length = len(sent_tokens)
                    
                    if current_length + sent_length > max_chunk_size and current_chunk:
                        # Store current chunk
                        chunk_text = ' '.join(current_chunk)
                        if len(self.tokenizer.encode(chunk_text)) >= min_chunk_size:
                            chunks.append({
                                'text': chunk_text,
                                'token_count': current_length
                            })
                        current_chunk = []
                        current_length = 0
                    
                    current_chunk.append(sent)
                    current_length += sent_length
            else:
                # Handle paragraph as a unit if possible
                if current_length + para_length > max_chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(self.tokenizer.encode(chunk_text)) >= min_chunk_size:
                        chunks.append({
                            'text': chunk_text,
                            'token_count': current_length
                        })
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(para)
                current_length += para_length
        
        # Handle remaining text
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(self.tokenizer.encode(chunk_text)) >= min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'token_count': current_length
                })
        
        return chunks


# Dependency to get the current user
def get_user(request: Request):
    """Validate and get user information."""

    if "user_access_token" in request.session:
        token = request.session["user_access_token"]
    else:
        token = None
        return None
    if token:
        try:
            manager = KadiManager(instance=instance, host=host, pat=token)
            user = manager.pat_user
            return user.meta["displayname"]
        except kadi_apy.lib.exceptions.KadiAPYRequestError as e:
            print(e)
            return None
    return None  # "Authed but Failed at getting user info!"


@app.get("/")
def public(request: Request, user=Depends(get_user)):
    """Main extrance of app."""

    root_url = gr.route_utils.get_root_url(request, "/", None)
    # print("root url", root_url)
    if user:
        return RedirectResponse(url=f"{root_url}/gradio/")
    else:
        return RedirectResponse(url=f"{root_url}/main/")


# Logout
@app.route("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    request.session.pop("user_id", None)
    request.session.pop("user_access_token", None)

    return RedirectResponse(url="/")


# Login
@app.route("/login")
async def login(request: Request):
    root_url = gr.route_utils.get_root_url(request, "/login", None)
    redirect_uri = request.url_for("auth")  # f"{root_url}/auth"
    redirect_uri = redirect_uri.replace(scheme="https")  # required by Kadi
    # print("-----------in login")
    # print("root_urlt", root_url)
    # print("redirect_uri", redirect_uri)
    # print("request", request)
    return await oauth.kadi4mat.authorize_redirect(request, redirect_uri)


# Get auth
@app.route("/auth")
async def auth(request: Request):
    root_url = gr.route_utils.get_root_url(request, "/auth", None)
    try:
        access_token = await oauth.kadi4mat.authorize_access_token(request)
        request.session["user_access_token"] = access_token["access_token"]

        # Redirect to Gradio app immediately after authorization
        return RedirectResponse(url="/gradio")

    except OAuthError as e:
        print("Error getting access token", e)
        return RedirectResponse(url="/")


def load_pdf(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return {"source": Path(file_path).name, "content": text, "metadata": {"file_type": "pdf"}}

def load_docx(file_path):
    """Extract text from a DOCX file."""
    doc = DocxReader(file_path)
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return {"source": Path(file_path).name, "content": text, "metadata": {"file_type": "docx"}}

def load_excel(file_path):
    """Extract text from an Excel file."""
    df = pd.read_excel(file_path)
    text = df.to_string(index=False)
    return {"source": Path(file_path).name, "content": text, "metadata": {"file_type": "excel"}}

def load_text(file_path):
    """Read plain text from a TXT file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return {"source": Path(file_path).name, "content": text, "metadata": {"file_type": "txt"}}

def load_markdown(file_path):
    """Convert Markdown content to plain text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return {"source": Path(file_path).name, "content": text, "metadata": {"file_type": "md"}}

def load_csv(file_path):
    """Extract text from a CSV file."""
    df = pd.read_csv(file_path)
    text = df.to_string(index=False)
    return {"source": Path(file_path).name, "content": text, "metadata": {"file_type": "csv"}}

def load_file(file_path):
    """Unified loader for different file types."""
    file_type = detect_file_type(file_path)
    if file_type == "pdf":
        return load_pdf(file_path)
    elif file_type == "docx":
        return load_docx(file_path)
    elif file_type == "excel":
        return load_excel(file_path)
    elif file_type == "txt":
        return load_text(file_path)
    elif file_type == "md":
        return load_markdown(file_path)
    elif file_type == "csv":
        return load_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def detect_file_type(file_path):
    """Detect file type using mimetypes and file extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    ext = Path(file_path).suffix.lower()
    if mime_type == "application/pdf" or ext == ".pdf":
        return "pdf"
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or ext == ".docx":
        return "docx"
    elif mime_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"] or ext in [".xls", ".xlsx"]:
        return "excel"
    elif mime_type == "text/plain" or ext == ".txt":
        return "txt"
    elif mime_type == "text/markdown" or ext == ".md":
        return "md"
    elif mime_type == "text/csv" or ext == ".csv":
        return "csv"
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

async def process_all_records_and_files(user_token, progress=gr.Progress()):
    """Process all records and their files."""
    try:
        manager = KadiManager(instance=instance, host=host, pat=user_token)
        all_records = get_all_records(user_token)
        
        rag = EnhancedRAG()
        total_chunks = 0
        
        for i, record_id in enumerate(all_records):
            try:
                record = manager.record(identifier=record_id)
                metadata = {
                    'record_id': record_id,
                    'title': record.meta.get('title', ''),
                    'description': record.meta.get('description', '')
                }
                
                # Get all files in record
                files = record.get_filelist().json()['items']
                
                for file_info in files:
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            file_path = os.path.join(temp_dir, file_info['name'])
                            record.download_file(file_info['id'], file_path)
                            
                            # Load and process file
                            doc = load_file(file_path)
                            chunks_added = rag.add_documents([doc], metadata)
                            total_chunks += chunks_added
                            
                    except Exception as e:
                        print(f"Error processing file {file_info['name']}: {str(e)}")
                        continue
                        
                progress((i + 1) / len(all_records))
                
            except Exception as e:
                print(f"Error processing record {record_id}: {str(e)}")
                continue
                
        return rag, total_chunks
        
    except Exception as e:
        print(f"Error in process_all_records_and_files: {str(e)}")
        return None, 0

def greet(request: gr.Request):
    """Show greeting message."""

    return f"Welcome to Kadichat, you're logged in as: {request.username}"


def get_files_in_record(record_id, user_token, top_k=10):
    """Get all file list within one record."""

    manager = KadiManager(instance=instance, host=host, pat=user_token)

    try:
        record = manager.record(identifier=record_id)
    except kadi_apy.lib.exceptions.KadiAPYInputError as e:
        raise gr.Error(e)

    file_num = record.get_number_files()

    per_page = 100  # default in kadi
    not_divisible = file_num % per_page
    if not_divisible:
        page_num = file_num // per_page + 1
    else:
        page_num = file_num // per_page

    file_names = []
    for p in range(1, page_num + 1):  # page starts at 1 in kadi
        file_names.extend(
            [
                info["name"]
                for info in record.get_filelist(page=p, per_page=per_page).json()[
                    "items"
                ]
            ]
        )

    assert file_num == len(
        file_names
    ), "Number of files did not match, please check function get_all_file_names."

    # return file_names[:top_k]
    return gr.Dropdown(
        choices=file_names[:top_k],
        label="Select file",
        info="Select (max. 3) files to chat with.",
        multiselect=True,
        max_choices=3,
        interactive=True,
    )


def get_all_records(user_token):
    """Get all record list in Kadi."""

    if not user_token:
        return []

    manager = KadiManager(instance=instance, host=host, pat=user_token)

    host_api = manager.host if manager.host.endswith("/") else manager.host + "/"
    searched_resource = "records"
    endpoint = urljoin(
        host_api, searched_resource
    )  # e.g https://demo-kadi4mat.iam.kit.edu/api/" + "records"

    response = manager.search.search_resources("record", per_page=100)
    parsed = json.loads(response.content)

    total_pages = parsed["_pagination"]["total_pages"]

    def get_page_records(parsed_content):
        item_identifiers = []
        items = parsed_content["items"]
        for item in items:
            item_identifiers.append(item["identifier"])

        return item_identifiers

    all_records_identifiers = []
    for page in range(1, total_pages + 1):
        page_endpoint = endpoint + f"?page={page}&per_page=100"
        response = manager.make_request(page_endpoint)
        parsed = json.loads(response.content)
        all_records_identifiers.extend(get_page_records(parsed))

    return all_records_identifiers


def _init_user_token(request: gr.Request):
    """Init user token."""

    user_token = request.request.session["user_access_token"]
    return user_token


# Landing page for login
with gr.Blocks() as login_demo:
    gr.Markdown(
        """<br/><br/><br/><br/><br/><br/><br/><br/>
            <center>
            <h1>Welcome to KadiChat!</h1>
            <br/><br/>
            <img src="https://i.postimg.cc/qvsQCCLS/kadichat-logo.png" alt="Kadichat logo">
            <br/><br/>
            Chat with Record in Kadi.</center>
            """
    )
    # Note: kadichat-logo is hosted on https://postimage.io/

    with gr.Row():
        with gr.Column():
            _btn_placeholder = gr.Button(visible=False)
        with gr.Column():
            btn = gr.Button("Sign in with Kadi (demo-instance)")
        with gr.Column():
            _btn_placeholder2 = gr.Button(visible=False)

    gr.Markdown(
        """<br/><br/><br/><br/>
            <center>
            This demo shows how to use
            <a href="https://kadi4mat.readthedocs.io/en/stable/httpapi/intro.html#oauth2-tokens">OAuth2</a> 
            to have access to Kadi.</center>
        """
    )
    _js_redirect = """
    () => {
        url = '/login' + window.location.search;
        window.open(url, '_blank');
    }
    """
    btn.click(None, js=_js_redirect)


# A simple RAG implementation
class EnhancedRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.vector_store = Chroma(
            collection_name="kadi_records",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"  # Add persistence
        )
        self.document_processor = DocumentProcessor()

    def add_documents(self, documents: List[dict], metadata: dict) -> int:
        """Add documents to vector store with metadata."""
        try:
            texts = []
            metadatas = []
            
            for doc in documents:
                # Simple preprocessing
                text = self.document_processor.preprocess_document(doc['content'])
                chunks = self.document_processor.chunk_document(text, max_chunk_size=512)
                
                for chunk in chunks:
                    texts.append(chunk['text'])
                    metadatas.append({
                        **metadata,
                        'source': doc.get('source', 'unknown'),
                        'chunk_size': chunk['token_count']
                    })
            
            if texts:
                self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            return len(texts)
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return 0

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Get relevant context for query."""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k*2)  # Get more candidates
            return "\n".join(doc.page_content for doc, score in results if score < 1.0)[:1500]  # Stricter filtering
            
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return ""

def chat_response(message: str, history: List[Tuple[str, str]], rag_system: EnhancedRAG) -> Tuple[List[Tuple[str, str]], str]:
    """Generate chat response using RAG."""
    try:
        # Get relevant context
        context = rag_system.get_relevant_context(message)
        
        # Build a clearer prompt
        if context:
            prompt = f"""<s>[INST] You are a helpful assistant answering questions about documents. Use the following context to answer the question, but only if it's relevant. If the context doesn't help answer the question, just answer naturally.

Context:
{context}

Question: {message}

Please provide a clear, direct answer. [/INST]"""
        else:
            prompt = f"""<s>[INST] You are a helpful assistant. Please answer this question:

{message} [/INST]"""

        # Get response from LLM
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=1024
"""             temperature=0.3,  # Lower temperature for more focused responses
            repetition_penalty=1.1,
            do_sample=True,
            top_p=0.9 """
        )
        
        # Improved response cleaning
        response = response.replace("<s>", "").replace("</s>", "")
        response = response.split("[/INST]")[-1].strip()
        
        # Additional cleaning
        response = re.sub(r'={3,}.*?={3,}', '', response)  # Remove === sections
        response = re.sub(r'\[.*?\]', '', response)  # Remove [...] sections
        response = re.sub(r'\n\s*\n+', '\n\n', response)  # Clean up multiple newlines
        response = response.strip()
        
        # Validate response
        if not response or len(response) < 10 or '=====' in response:
            return history + [(message, "I apologize, but I couldn't generate a proper response. Please try asking your question again.")], ""
            
        return history + [(message, response)], ""
        
    except Exception as e:
        print(f"Error in chat response: {str(e)}")
        return history + [(message, "I encountered an error. Please try again.")], ""

def process_chat(message, history, loading_state):
    """Process chat messages and return responses."""
    if loading_state:
        return history + [(message, "Still loading documents. Please wait...")], ""
    
    # Get the RAG system from the state
    rag_system = user_session_rag.value
    
    if not rag_system or not hasattr(rag_system, 'vector_store'):
        return history + [(message, "RAG system not properly initialized. Please refresh the page.")], ""
    
    return chat_response(message, history, rag_system)

def handle_chat(message, history, loading_state):
    """Handle chat with loading state and enhanced response."""
    try:
        if loading_state:
            return history + [(message, "Still loading documents. Please wait...")], ""
            
        # Get the RAG system from the state
        rag_system = user_session_rag.value
        
        if not rag_system or not hasattr(rag_system, 'vector_store'):
            return history + [(message, "RAG system not properly initialized. Please refresh the page.")], ""
            
        return chat_response(message, history, rag_system)
        
    except Exception as e:
        error_msg = f"Error in chat handler: {str(e)}"
        print(error_msg)
        return history + [(message, "An error occurred. Please try again.")], ""


app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app = gr.mount_gradio_app(app, login_demo, path="/main")

# Gradio interface
with gr.Blocks() as main_demo:
    # State variables
    _state_user_token = gr.State([])
    user_session_rag = gr.State(None)  # Initialize as None instead of SimpleRAG()
    loading_state = gr.State(True)

    # Header with welcome message and logout button
    with gr.Row():
        with gr.Column(scale=7):
            welcome_msg = gr.Markdown("Welcome to Chatbot!")
            main_demo.load(greet, None, welcome_msg)
        with gr.Column(scale=1):
            gr.Button("Logout", link="/logout")

    loading_status = gr.Markdown("Initializing system...")

    async def initialize_system(request: gr.Request):
        try:
            user_token = request.request.session["user_access_token"]
            rag_system, num_chunks = await process_all_records_and_files(user_token)
            
            # Update the global state
            user_session_rag.value = rag_system
            loading_state.value = False
            
            # Show both document count and chunk count
            doc_count = len(rag_system.vector_store._collection.count())  # Get actual document count
            return f"System Status: Ready! Processed {doc_count} documents into {num_chunks} searchable chunks."
        except Exception as e:
            loading_state.value = False
            return f"System Status: Error - {str(e)}"

    # Update loading status when initialization completes
    main_demo.load(
        fn=initialize_system,
        inputs=None,
        outputs=loading_status,
    )

    # Main chat interface
    with gr.Tab("Chat"):
        with gr.Row():
            # Chat display
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(height=600)
                
            # System status (optional)
            with gr.Column(scale=3):
                system_status = gr.Markdown("System Status: Initializing...")

        # Input area
        with gr.Row():
            txt_input = gr.Textbox(
                show_label=False,
                placeholder="Type your question here...",
                lines=2,
                scale=8
            )
            
        # Buttons
        with gr.Row():
            submit_btn = gr.Button("Submit", scale=2)
            refresh_btn = gr.Button("Clear Chat", scale=1, variant="secondary")

        # Example questions
        gr.Examples(
            examples=[
                ["Summarize the contents of my records."],
                ["What are the main topics discussed in my documents?"],
                ["How many records do I have in total?"],
            ],
            inputs=[txt_input]
        )

        # Define chat functionality
        txt_input.submit(
            fn=process_chat,
            inputs=[txt_input, chatbot, loading_state],
            outputs=[chatbot, txt_input]
        )
        
        submit_btn.click(
            fn=process_chat,
            inputs=[txt_input, chatbot, loading_state],
            outputs=[chatbot, txt_input]
        )
        
        refresh_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, txt_input]
        )

        # Update system status periodically
        def update_status(loading_state):
            if loading_state:
                return "System Status: Loading documents..."
            return "System Status: Ready"

        main_demo.load(
            fn=update_status,
            inputs=[loading_state],
            outputs=[system_status],
            every=1
        )

app = gr.mount_gradio_app(app, main_demo, path="/gradio", auth_dependency=get_user)

if __name__ == "__main__":
    uvicorn.run(app, port=7860, host="0.0.0.0")
    