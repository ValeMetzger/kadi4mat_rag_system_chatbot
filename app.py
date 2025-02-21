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
from typing import List, Tuple, Optional
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
import nltk
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
#from langchain_groq import ChatGroq


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
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            # Use sent_tokenize directly instead of loading the tokenizer
            self.nltk_tokenizer = nltk.sent_tokenize
        except Exception as e:
            print(f"Error initializing NLTK: {str(e)}")
            self.nltk_tokenizer = None
    
    def preprocess_document(self, text: str) -> str:
        """Clean and preprocess document text."""
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def chunk_document(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split document into semantic chunks."""
        try:
            if self.nltk_tokenizer:
                sentences = self.nltk_tokenizer(text)
            else:
                # Fallback to basic sentence splitting
                sentences = re.split('[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                # Get token count for the sentence
                tokens = self.tokenizer.encode(sentence)
                sentence_length = len(tokens)
                
                if current_length + sentence_length > max_chunk_size:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
            
        except Exception as e:
            print(f"Error in chunk_document: {str(e)}")
            # Return single chunk if chunking fails
            return [text]


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
    print(f"Found {total_records} records to process")
    progress(0, desc=f"Found {total_records} records to process")

    for i, record_id in enumerate(all_records):
        try:
            record = manager.record(identifier=record_id)
            file_names = [info["name"] for info in record.get_filelist().json()["items"]]
            print(f"Processing record {record_id} with {len(file_names)} files")
            
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
                        print(f"Successfully processed file: {file_name}")
                except Exception as e:
                    print(f"Error processing file {file_name} in record {record_id}: {str(e)}")

            progress((i + 1) / total_records, desc=f"Processing record {i + 1}/{total_records}")
        except Exception as e:
            print(f"Error processing record {record_id}: {str(e)}")
            continue

    print(f"\nProcessed {len(documents)} documents in total")
    
    # Initialize RAG system with all documents
    rag = SimpleRAG()
    rag.documents = documents
    print("\nBuilding vector database...")
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
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.vector_store = None
        self.documents = []
    
    def search_documents(self, query: str, k: int = 3, doc_filter: Optional[List[str]] = None) -> List[str]:
        """
        Search for relevant document chunks.
        
        Args:
            query: The search query
            k: Number of results to return
            doc_filter: Optional list of document IDs or names to filter by
        """
        try:
            if not self.vector_store:
                print("Warning: Vector store not initialized")
                return []
            
            print(f"\nSearching for query: {query}")
            
            # Search with scores
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k
            )
            
            print(f"Found {len(results)} results")
            
            filtered_results = []
            seen_doc_ids = set()
            
            for idx, (doc, score) in enumerate(results):
                print(f"\nResult {idx}:")
                print(f"Score: {score}")
                print(f"Content preview: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
                
                # Check if document matches filter criteria
                if doc_filter:
                    doc_id = doc.metadata.get('source', '')
                    if not any(f in doc_id for f in doc_filter):
                        print(f"Skipped due to document filter")
                        continue
                
                # Apply score threshold
                if score < 2.0:
                    doc_id = doc.metadata.get('doc_id')
                    if doc_id not in seen_doc_ids:
                        filtered_results.append({
                            'content': doc.page_content,
                            'source': doc.metadata.get('source', 'unknown'),
                            'score': score
                        })
                        seen_doc_ids.add(doc_id)
                        print(f"Added to filtered results (doc_id: {doc_id})")
                else:
                    print("Skipped due to score threshold")
            
            print(f"\nReturning {len(filtered_results)} filtered results")
            return filtered_results
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []

    def get_context_for_query(self, query: str, doc_filter: Optional[List[str]] = None) -> str:
        """
        Get formatted context for a query, optionally filtered by specific documents.
        """
        try:
            relevant_chunks = self.search_documents(query, doc_filter=doc_filter)
            if not relevant_chunks:
                return ""
            
            # Format context with source information
            formatted_chunks = []
            for chunk in relevant_chunks:
                formatted_chunks.append(f"""
Source: {chunk['source']}
---
{chunk['content']}
""")
            
            context = "\n\n".join(formatted_chunks)
            
            # Limit context length if needed
            if len(context) > 3000:  # Increased limit
                context = context[:2997] + "..."
            
            return context
            
        except Exception as e:
            print(f"Error getting context: {str(e)}")
            return ""

    def build_vector_db(self):
        """Initialize or rebuild the vector database with current documents."""
        try:
            from langchain_community.vectorstores import Chroma
            
            print("\nStarting vector database build...")
            print(f"Number of documents to process: {len(self.documents)}")
            
            # Debug document contents
            for idx, doc in enumerate(self.documents):
                print(f"\nDocument {idx}:")
                print(f"Source: {doc.get('source', 'unknown')}")
                print(f"Content length: {len(doc.get('content', ''))}")
                print(f"Content preview: {doc.get('content', '')[:200]}...")
            
            processed_chunks = []
            chunk_metadata = []
            
            for doc_idx, doc in enumerate(self.documents):
                clean_text = self.document_processor.preprocess_document(doc['content'])
                chunks = self.document_processor.chunk_document(clean_text)
                print(f"\nDocument {doc_idx} produced {len(chunks)} chunks")
                
                for chunk_idx, chunk in enumerate(chunks):
                    processed_chunks.append(chunk)
                    chunk_metadata.append({
                        "chunk_id": len(processed_chunks) - 1,
                        "doc_id": f"doc_{doc_idx}",
                        "source": doc.get('source', 'unknown'),
                        "file_type": doc.get('metadata', {}).get('file_type', 'unknown')
                    })
            
            print(f"\nTotal chunks created: {len(processed_chunks)}")
            if processed_chunks:
                print(f"First chunk preview: {processed_chunks[0][:200]}...")
            
            if not processed_chunks:
                print("Warning: No chunks were created from the documents")
                return 0
            
            # Initialize vector store
            self.vector_store = Chroma(
                collection_name="documents",
                embedding_function=self.embeddings
            )
            
            # Add chunks with their corresponding metadata
            self.vector_store.add_texts(
                texts=processed_chunks,
                metadatas=chunk_metadata
            )
            
            print(f"\nSuccessfully built vector database with {len(processed_chunks)} chunks")
            return len(processed_chunks)
            
        except Exception as e:
            print(f"Error building vector database: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    def add_documents(self, documents: List[str]):
        """Process and add documents to the RAG system."""
        try:
            processed_chunks = []
            for doc in documents:
                # Preprocess and chunk the document
                clean_text = self.document_processor.preprocess_document(doc)
                chunks = self.document_processor.chunk_document(clean_text)
                processed_chunks.extend(chunks)
            
            # Store original documents
            self.documents = documents
            
            # Initialize or update vector store
            if not self.vector_store:
                self.vector_store = Chroma(
                    collection_name="documents",
                    embedding_function=self.embeddings
                )
            
            # Add documents with metadata
            self.vector_store.add_texts(
                texts=processed_chunks,
                metadatas=[{
                    "chunk_id": i,
                    "doc_id": f"doc_{i//5}"  # Assuming ~5 chunks per doc
                } for i in range(len(processed_chunks))]
            )
            
            return len(processed_chunks)
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return 0
    
    def clear_documents(self):
        """Clear all documents from the RAG system."""
        try:
            if self.vector_store:
                self.vector_store.delete_collection()
                self.vector_store = None
            self.documents = []
        except Exception as e:
            print(f"Error clearing documents: {str(e)}")
    
    def get_document_count(self) -> int:
        """Get the number of original documents."""
        return len(self.documents)
    
    def is_initialized(self) -> bool:
        """Check if the RAG system is properly initialized."""
        return self.vector_store is not None and len(self.documents) > 0




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


def clean_response(response: str) -> str:
    """Clean up model response while preserving content."""
    try:
        # Remove any system prompts or artifacts
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1]
        
        # Remove any remaining special tokens
        response = response.replace("<s>", "").replace("</s>", "")
        response = response.replace("[INST]", "").strip()
        
        # Only clean up excessive whitespace
        response = "\n".join(line.strip() for line in response.split("\n"))
        
        # Return the response if it has any content
        if response.strip():
            return response
        return "I apologize, but I couldn't generate a response. Please try again."
        
    except Exception as e:
        print(f"Error cleaning response: {str(e)}")
        return str(e)


def respond(message: str, history: List[Tuple[str, str]], user_session_rag):
    try:
        # Get context from RAG system
        context = user_session_rag.get_context_for_query(message)
        
        # Construct a more natural prompt
        prompt_parts = [
            "<s>[INST] You are a knowledgeable AI assistant having a natural conversation. You have access to both general knowledge and specific documents.\n\n"
        ]
        
        if context:
            prompt_parts.append(f"Here are some relevant passages from the documents I have access to:\n\n{context}\n\n")
        
        prompt_parts.extend([
            f"The user asks: {message}\n\n",
            "Please provide a natural, informative response. You can:\n",
            "- Draw from both the documents and your general knowledge\n",
            "- Freely explore related topics that might be interesting or relevant\n",
            "- Reference specific documents when they're relevant\n",
            "- Acknowledge when you're speculating or going beyond the documents\n",
            "- Use a conversational, engaging tone\n\n",
            "Remember: You're not limited to only discussing the documents - use them when relevant but feel free to expand the discussion naturally. [/INST]"
        ])
        
        prompt = "".join(prompt_parts)

        # Generate response with creative parameters
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=2048,  # Allow for longer responses
            temperature=0.8,  # More creative
            top_p=0.95,  # More diverse vocabulary
            repetition_penalty=1.05,  # Allow more natural repetition
            do_sample=True,
            stop_sequences=["</s>"]  # Only stop at end of sequence
        )
        
        # Simpler response cleaning
        cleaned_response = response.split("[/INST]")[-1].strip()
        cleaned_response = cleaned_response.replace("<s>", "").replace("</s>", "")
        
        return history + [(message, cleaned_response)], ""
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return history + [(message, error_msg)], ""


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
    