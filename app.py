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
            manager = KadiManager(instance=instance, host=host, token=token)
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
    for i, record_id in enumerate(all_records):
        record = manager.record(identifier=record_id)
        file_names = [info["name"] for info in record.get_filelist().json()["items"]]
        for file_name in file_names:
            file_id = record.get_file_id(file_name)
            with tempfile.TemporaryDirectory(prefix="tmp-kadichat-downloads-") as temp_dir:
                temp_file_location = os.path.join(temp_dir, file_name)
                record.download_file(file_id, temp_file_location)
                try:
                    doc = load_file(temp_file_location)
                    documents.append(doc)
                except ValueError as e:
                    print(f"Skipping unsupported file type: {file_name}")

        progress((i + 1) / total_records, desc=f"Processing record {i + 1}/{total_records}")

    user_rag = SimpleRAG()
    user_rag.documents = documents
    user_rag.embeddings_model = embeddings_model
    user_rag.build_vector_db()
    global user_session_rag
    user_session_rag = user_rag

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
        self.embeddings = None
        self.index = None
        # self.load_pdf("Brandt et al_2024_Kadi_info_page.pdf")
        # self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the property documents by page."""

        doc = pymupdf.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the documents."""
        if self.embeddings_model is None:
            self.embeddings_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2", trust_remote_code=True
            )

        # Ensure self.documents is a list of dictionaries with "content" key
        valid_documents = [doc for doc in self.documents if isinstance(doc, dict) and "content" in doc]
        if len(valid_documents) != len(self.documents):
            print("Warning: Some documents are missing the 'content' key or are not dictionaries.")

        texts = [doc["content"] for doc in valid_documents]
        embeddings = self.embeddings_model.encode(texts, show_progress_bar=True)
        self.vector_db = chromadb.Client()
        self.vector_db.add_documents(texts, embeddings)
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 4) -> List[str]:
        """Searches for relevant documents using vector similarity."""

        # Use embeddings_client
        # query_embedding = self.embeddings_model.encode([query], show_progress_bar=False)
        embedding_responses = embeddings_client.post(
            json={"inputs": [query]}, task="feature-extraction"
        )
        query_embedding = json.loads(embedding_responses.decode())
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]


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
    """Parse file and prepare RAG."""

    if not file_names:
        raise gr.Error("No file selected")
    progress(0, desc="Starting")
    # Create connection to kadi
    manager = KadiManager(instance=instance, host=host, pat=token)
    record = manager.record(identifier=record_id)
    progress(0.2, desc="Loading files...")
    # Parse files
    documents = []
    # Download
    for file_name in file_names:
        file_id = record.get_file_id(file_name)
        with tempfile.TemporaryDirectory(prefix="tmp-kadichat-downloads-") as temp_dir:
            print(temp_dir)
            temp_file_location = os.path.join(temp_dir, file_name)
            record.download_file(file_id, temp_file_location)
            # parse document
            docs = load_and_chunk_pdf(temp_file_location)
            documents.extend(docs)

    progress(0.4, desc="Embedding documents...")
    user_rag = SimpleRAG()
    user_rag.documents = documents
    user_rag.embeddings_model = embeddings_model
    user_rag.build_vector_db()
    # print(documents[:2])
    print("user rag created")
    progress(1, desc="ready to chat")
    return "ready to chat", user_rag


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

    # message is the current input query from user
    # RAG
    retrieved_docs = user_session_rag.search_documents(message)
    context = "\n".join(retrieved_docs)
    system_message = "You are an assistant to help user to answer question related to Kadi based on Relevant documents.\nRelevant documents: {}".format(
        context
    )
    messages = [{"role": "assistant", "content": system_message}]

    # Add history for conversational chat, TODO
    # for val in history:
    #     #if val[0]:
    #     messages.append({"role": "user", "content": val[0]})
    #     #if val[1]:
    #     messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": f"\nQuestion: {message}"})

    # print("-----------------")
    # print(messages)
    # print("-----------------")
    # Get anwser from LLM
    response = client.chat_completion(
        messages, max_tokens=2048, temperature=0.0
    )  # , top_p=0.9)
    response_content = "".join(
        [
            choice.message["content"]
            for choice in response.choices
            if "content" in choice.message
        ]
    )

    # Process response
    polished_response = preprocess_response(response_content)

    history.append((message, polished_response))
    return history, ""


app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app = gr.mount_gradio_app(app, login_demo, path="/main")

# Gradio interface
with gr.Blocks() as main_demo:

    # State for storing user token
    _state_user_token = gr.State([])

    # State for user rag
    user_session_rag = gr.State(SimpleRAG())  # Initialize as SimpleRAG instance

    with gr.Row():
        with gr.Column(scale=7):
            m = gr.Markdown("Welcome to Chatbot!")
            main_demo.load(greet, None, m)
        with gr.Column(scale=1):
            gr.Button("Logout", link="/logout")

    with gr.Tab("Main"):
        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot()

        with gr.Row():
            txt_input = gr.Textbox(
                show_label=False, placeholder="Type your question here...", lines=1
            )
            submit_btn = gr.Button("Submit", scale=1)
            refresh_btn = gr.Button("Refresh Chat", scale=1, variant="secondary")

        example_questions = [
            ["Summarize the paper."],
            ["how to create record in kadi4mat?"],
        ]

        gr.Examples(examples=example_questions, inputs=[txt_input])

        # Actions
        txt_input.submit(
            fn=respond,
            inputs=[txt_input, chatbot, user_session_rag],
            outputs=[chatbot, txt_input],
        )
        submit_btn.click(
            fn=respond,
            inputs=[txt_input, chatbot, user_session_rag],
            outputs=[chatbot, txt_input],
        )
        refresh_btn.click(lambda: [], None, chatbot)

app = gr.mount_gradio_app(app, main_demo, path="/gradio", auth_dependency=get_user)

if __name__ == "__main__":
    uvicorn.run(app, port=7860, host="0.0.0.0")