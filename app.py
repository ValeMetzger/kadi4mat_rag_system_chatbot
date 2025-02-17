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
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import mimetypes
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document as DocxReader
import pandas as pd
import chromadb

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
from huggingface_hub import InferenceClient
from langchain_groq import ChatGroq

client = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=GROQ_API_KEY)

# Mixed-usage of huggingface client and local model for showing 2 possibilities
embeddings_client = InferenceClient(
    model="sentence-transformers/all-mpnet-base-v2", token=huggingfacehub_api_token
)
embeddings_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2", trust_remote_code=True
)


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
    """Process all records and files for the user."""
    manager = KadiManager(instance=instance, host=host, pat=user_token)
    all_records = get_all_records(user_token)

    documents = []
    total_records = len(all_records)
    progress(0, desc=f"Found {total_records} records to process")

    for i, record_id in enumerate(all_records):
        try:
            record = manager.record(identifier=record_id)
            file_names = [info["name"] for info in record.get_filelist().json()["items"]]
            
            for file_name in file_names:
                try:
                    file_id = record.get_file_id(file_name)
                    with tempfile.TemporaryDirectory(prefix="tmp-kadichat-downloads-") as temp_dir:
                        temp_file_location = os.path.join(temp_dir, file_name)
                        record.download_file(file_id, temp_file_location)
                        doc = load_file(temp_file_location)
                        # Add record metadata to the document
                        doc["metadata"]["record_id"] = record_id
                        documents.append(doc)
                except Exception as e:
                    print(f"Error processing file {file_name} in record {record_id}: {str(e)}")

            progress((i + 1) / total_records, desc=f"Processing record {i + 1}/{total_records}")
        except Exception as e:
            print(f"Error processing record {record_id}: {str(e)}")
            continue

    # Initialize RAG system with all documents
    rag = SimpleRAG()
    rag.documents = documents
    rag.embeddings_model = embeddings_model
    print(f"Building vector database with {len(documents)} documents...")
    rag.build_vector_db()
    
    # Update the global RAG system
    global user_session_rag
    user_session_rag = rag
    
    return len(documents)

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
class SimpleRAG:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings_model = None
        self.index = None
        self.collection = None
        self.is_initialized = False  # Add a flag to track initialization

    def build_vector_db(self) -> None:
        """
        Builds a vector database using the content of the documents.
        Includes error handling, logging, and document validation.
        """
        print("\n=== Starting Vector Database Build Process ===")
        
        # Validate embeddings model
        if self.embeddings_model is None:
            print("Initializing embeddings model...")
            try:
                self.embeddings_model = SentenceTransformer(
                    "sentence-transformers/all-mpnet-base-v2",
                    trust_remote_code=True
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize embeddings model: {str(e)}")

        # Validate documents
        if not self.documents:
            raise ValueError("No documents available to build the vector database.")
        
        print(f"Processing {len(self.documents)} documents...")

        # Validate document structure and content
        valid_documents = []
        for i, doc in enumerate(self.documents):
            try:
                if not isinstance(doc, dict):
                    print(f"Skipping document {i}: Not a dictionary")
                    continue
                if "content" not in doc:
                    print(f"Skipping document {i}: No content field")
                    continue
                if not doc["content"] or not isinstance(doc["content"], str):
                    print(f"Skipping document {i}: Invalid content")
                    continue
                if len(doc["content"].strip()) == 0:
                    print(f"Skipping document {i}: Empty content")
                    continue
                valid_documents.append(doc)
            except Exception as e:
                print(f"Error processing document {i}: {str(e)}")
                continue

        if not valid_documents:
            raise ValueError("No valid documents found after validation")

        print(f"Found {len(valid_documents)} valid documents")

        try:
            # Initialize ChromaDB with persistent storage
            print("Initializing ChromaDB...")
            persist_directory = "./chroma_db"
            os.makedirs(persist_directory, exist_ok=True)
            
            self.index = chromadb.PersistentClient(path=persist_directory)
            collection_name = "documents"

            # Clean up existing collection if it exists
            try:
                self.index.delete_collection(collection_name)
                print("Deleted existing collection")
            except Exception as e:
                print(f"No existing collection to delete: {str(e)}")

            # Create new collection
            self.collection = self.index.create_collection(
                name=collection_name,
                metadata={"description": "Document collection for RAG system"}
            )
            print("Created new collection")

            # Prepare documents and metadata
            texts = []
            metadatas = []
            ids = []
            
            for idx, doc in enumerate(valid_documents):
                texts.append(doc["content"])
                metadata = doc.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {"original_metadata": str(metadata)}
                metadata["doc_id"] = str(idx)
                metadata["source"] = doc.get("source", "unknown")
                metadatas.append(metadata)
                ids.append(f"doc_{idx}")

            # Generate embeddings
            print("Generating embeddings...")
            try:
                embeddings = self.embeddings_model.encode(
                    texts,
                    show_progress_bar=True,
                    batch_size=32,
                    normalize_embeddings=True
                )
                print(f"Generated {len(embeddings)} embeddings")
            except Exception as e:
                raise ValueError(f"Failed to generate embeddings: {str(e)}")

            # Validate embeddings
            if len(embeddings) != len(texts):
                raise ValueError(f"Embedding count ({len(embeddings)}) doesn't match document count ({len(texts)})")

            # Add documents to collection
            print("Adding documents to collection...")
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            # Verify collection size
            collection_size = self.collection.count()
            if collection_size != len(valid_documents):
                print(f"Warning: Collection size ({collection_size}) differs from input document count ({len(valid_documents)})")
            
            print(f"Successfully built vector database with {collection_size} documents")
            print("=== Vector Database Build Process Complete ===\n")

        except Exception as e:
            error_msg = f"Failed to build vector database: {str(e)}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)

    def load_file(self, file_path: str) -> None:
        """Loads a file and stores its content in the property documents."""
        file_type = detect_file_type(file_path)
        print(f"Loading {file_type} file: {file_path}")
        
        try:
            if file_type == "pdf":
                self.documents = load_and_chunk_pdf(file_path)
            elif file_type == "docx":
                self.documents = [load_docx(file_path)]
            elif file_type == "excel":
                self.documents = [load_excel(file_path)]
            elif file_type == "txt":
                self.documents = [load_text(file_path)]
            elif file_type == "md":
                self.documents = [load_markdown(file_path)]
            elif file_type == "csv":
                self.documents = [load_csv(file_path)]
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            if not self.documents:
                raise ValueError(f"No content extracted from file: {file_path}")
                
            print(f"Successfully loaded {len(self.documents)} document segments")
        except Exception as e:
            raise ValueError(f"Failed to load file {file_path}: {str(e)}")
        

    def search_documents(self, query: str, k: int = 4) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        if not self.collection:
            print("Warning: No collection available for search")
            return ["Please load documents first."]
        
        try:
            query_embedding = embeddings_client.feature_extraction([query])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            return results['documents'][0] if results['documents'] else ["No relevant documents found."]
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return [f"Search failed: {str(e)}"]




def chunk_text(text, chunk_size=2048, overlap_size=256, separators=["\n\n", "\n"]):
    """Chunk text into pieces of specified size with overlap, considering separators."""

    # Split the text by the separators
    for sep in separators:
        text = text.replace(sep, "\n")

    chunks = []
    start = 0

    while start < len(text):
        # Determine the end of the chunk, accounting for overlap and the chunk size
        end = min(len(text), start + chunk_size)

        # Find a natural break point at the newline to avoid cutting words
        if end < len(text):
            while end > start and text[end] != "\n":
                end -= 1

        chunk = text[start:end].strip()  # Strip trailing whitespace
        chunks.append(chunk)

        # Move the start position forward by the overlap size
        start += chunk_size - overlap_size

    return chunks


def load_and_chunk_pdf(file_path):
    """Extracts text from a PDF file and stores it in the property documents by chunks."""

    with pymupdf.open(file_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()

        chunks = chunk_text(text)
        documents = []
        for chunk in chunks:
            documents.append({"content": chunk, "metadata": pdf.metadata})

        return documents


def load_pdf(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return {"source": Path(file_path).name, "content": text, "metadata": {"file_type": "pdf"}}


def prepare_file_for_chat(record_id, file_names, token, progress=gr.Progress()):
    if not file_names:
        raise gr.Error("No file selected")
    
    progress(0, desc="Starting")
    manager = KadiManager(instance=instance, host=host, pat=token)
    record = manager.record(identifier=record_id)
    
    progress(0.2, desc="Loading files...")
    documents = []
    
    for file_name in file_names:
        try:
            file_id = record.get_file_id(file_name)
            with tempfile.TemporaryDirectory(prefix="tmp-kadichat-downloads-") as temp_dir:
                temp_file_location = os.path.join(temp_dir, file_name)
                record.download_file(file_id, temp_file_location)
                doc = load_file(temp_file_location)
                documents.append(doc)
        except Exception as e:
            print(f"Error processing file {file_name}: {str(e)}")

    # Create and initialize RAG system
    rag = SimpleRAG()
    rag.documents = documents
    rag.embeddings_model = embeddings_model
    rag.build_vector_db()
    
    progress(1, desc="Ready to chat")
    return "Ready to chat", rag


def preprocess_response(response: str) -> str:
    """Preprocesses the response to make it more polished."""

    # Placeholder for preprocessing

    # response = response.strip()
    # response = response.replace("\n\n", "\n")
    # response = response.replace(" ,", ",")
    # response = response.replace(" .", ".")
    # response = " ".join(response.split())
    # if not any(word in response.lower() for word in ["sorry", "apologize", "empathy"]):
    #     response = "I'm here to help. " + response
    return response


def respond(message: str, history: List[Tuple[str, str]], user_session_rag):
    """Get respond from LLMs."""
    
    if not user_session_rag or not hasattr(user_session_rag, 'documents') or not user_session_rag.documents:
        return (
            history + [
                (
                    message,
                    "Still loading documents or no documents found. Please wait a moment and try again.",
                )
            ],
            ""
        )

    try:
        # Get relevant documents
        retrieved_docs = user_session_rag.search_documents(message)
        context = "\n".join(retrieved_docs)
        
        # Prepare system message with context
        system_message = (
            "You are an assistant helping with Kadi documents. "
            "Here are the relevant documents to answer the question:\n\n"
            f"{context}"
        )
        
        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ]
        
        # Get response from LLM
        response = client.invoke(
            messages,
            max_tokens=2048,
            temperature=0.0
        )
        
        # Get the response content
        response_content = response.content  # Updated to access content directly
        
        # Update history and return
        history.append((message, response_content))
        return history, ""
        
    except Exception as e:
        error_message = f"Error processing your request: {str(e)}"
        history.append((message, error_message))
        return history, ""


app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app = gr.mount_gradio_app(app, login_demo, path="/main")

# Gradio interface
with gr.Blocks() as main_demo:
    # State variables
    _state_user_token = gr.State([])
    user_session_rag = gr.State(SimpleRAG())
    loading_state = gr.State(True)  # Track loading state

    # Header with welcome message and logout button
    with gr.Row():
        with gr.Column(scale=7):
            welcome_msg = gr.Markdown("Welcome to Chatbot!")
            main_demo.load(greet, None, welcome_msg)
        with gr.Column(scale=1):
            gr.Button("Logout", link="/logout")

    loading_status = gr.Markdown("Initializing system and loading records...")
    loading_state = gr.State(True)  # Start as True (loading)

    async def initialize_system(request: gr.Request):
        try:
            user_token = request.request.session["user_access_token"]
            num_docs = await process_all_records_and_files(user_token)
            loading_state.value = False
            return f"Ready to chat! Loaded {num_docs} documents from your records."
        except Exception as e:
            loading_state.value = False
            return f"Error loading records: {str(e)}"

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
                ["Find documents related to material science."],
                ["How many records do I have in total?"],
            ],
            inputs=[txt_input]
        )

        # Define chat functionality
        def chat_response(message, history, loading_state):
            if loading_state:
                return history + [(message, "Still loading documents. Please wait...")], ""
            return respond(message, history, user_session_rag)

        # Button actions
        txt_input.submit(
            fn=chat_response,
            inputs=[txt_input, chatbot, loading_state],
            outputs=[chatbot, txt_input]
        )
        
        submit_btn.click(
            fn=chat_response,
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