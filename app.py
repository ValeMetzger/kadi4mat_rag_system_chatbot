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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
from functools import wraps
import transformers
import torch

# Kadi OAuth settings
load_dotenv()
KADI_CLIENT_ID = os.environ["KADI_CLIENT_ID"]
KADI_CLIENT_SECRET = os.environ["KADI_CLIENT_SECRET"]
SECRET_KEY = os.environ["SECRET_KEY"]
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

# Initialize LLM components
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

try:
    # Initialize pipeline with bfloat16 and auto device mapping
    llm_pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL_ID,
        token=huggingfacehub_api_token,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=huggingfacehub_api_token,
        trust_remote_code=True
    )
    
except Exception as e:
    print(f"Error initializing LLaMA 3.1: {str(e)}")
    raise

# Mixed-usage of huggingface client and local model for showing 2 possibilities
embeddings_client = InferenceClient(
    model="sentence-transformers/all-mpnet-base-v2", token=huggingfacehub_api_token
)

# Add rate limiting decorator
def rate_limit(max_per_minute):
    """Rate limit decorator to prevent API overload."""
    interval = 60.0 / max_per_minute
    last_time = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            elapsed = current_time - last_time[0]
            if elapsed < interval:
                time.sleep(interval - elapsed)
            result = func(*args, **kwargs)
            last_time[0] = time.time()
            return result
        return wrapper
    return decorator

# Add tokenizer for token counting
def count_tokens(text):
    """Count tokens in text using the model's tokenizer."""
    return len(tokenizer.encode(text))

def validate_response(response_content: str) -> Tuple[bool, str]:
    """Enhanced validation of LLM responses with more robust corruption checks."""
    
    if not response_content or not isinstance(response_content, str):
        return False, "Invalid or empty response"
        
    # Check for common LLM artifacts that indicate corruption
    corruption_indicators = {
        'control_chars': ['\x00', '\x01', '\x02', '\x03', '\x04', '\x1a', '\x1b'],
        'incomplete_tags': ['<|', '|>', '<<', '>>', '{*', '*}'],
        'repeated_patterns': ['....', ',,,,', '????', '////'],
        'markdown_artifacts': ['```', '===', '***', '___'],
    }
    
    for category, indicators in corruption_indicators.items():
        for indicator in indicators:
            if indicator in response_content:
                return False, f"Contains {category}: {indicator}"
    
    # Check for coherent structure
    if len(response_content.strip()) < 10:
        return False, "Response too short"
        
    sentences = response_content.split('.')
    if len(sentences) > 0:
        first_sentence = sentences[0].strip()
        if not any(c.isalpha() for c in first_sentence):
            return False, "First sentence contains no letters"
            
    # Check for excessive repetition
    words = response_content.split()
    if len(words) >= 4:  # Only check if we have enough words
        repeated_sequences = 0
        for i in range(len(words) - 3):
            sequence = ' '.join(words[i:i+3])
            if response_content.count(sequence) > 2:
                repeated_sequences += 1
        if repeated_sequences > len(words) * 0.1:  # More than 10% repetition
            return False, "Contains excessive repetition"
            
    return True, "Valid response"

@rate_limit(max_per_minute=30)
def get_llm_response(messages: List[dict], max_retries: int = 3) -> str:
    """Enhanced LLM response handling using LLaMA 3.1 pipeline"""
    
    for attempt in range(max_retries):
        try:
            # Format messages into LLaMA 3.1 chat format
            formatted_prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted_prompt += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
                elif role == "user":
                    formatted_prompt += f"[INST] {content} [/INST]"
                elif role == "assistant":
                    formatted_prompt += f"{content} </s>"
            
            # Generate response
            outputs = llm_pipeline(
                formatted_prompt,
                max_new_tokens=1024,
                temperature=0.2 + (attempt * 0.1),  # Temperature jitter
                top_p=0.95,
                repetition_penalty=1.2,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Extract and clean response
            response_text = outputs[0]["generated_text"]
            
            # Remove the input prompt from the response
            response_content = response_text[len(formatted_prompt):].strip()
            
            # Remove any system or instruction tags
            response_content = response_content.replace("[INST]", "")
            response_content = response_content.replace("[/INST]", "")
            response_content = response_content.replace("<<SYS>>", "")
            response_content = response_content.replace("<</SYS>>", "")
            response_content = response_content.replace("</s>", "")
            
            # Validate response
            is_valid, reason = validate_response(response_content)
            if not is_valid:
                print(f"Attempt {attempt + 1} failed validation: {reason}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid response: {reason}")
                continue
            
            return preprocess_response(response_content)
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    
    raise RuntimeError("Failed to get valid response after all retries")

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
    # print("*****+ in auth")
    # print("root_urlt", root_url)
    # print("request", request)
    try:
        access_token = await oauth.kadi4mat.authorize_access_token(request)
        request.session["user_access_token"] = access_token["access_token"]

    except OAuthError as e:
        print("Error getting access token", e)
        return RedirectResponse(url="/")

    return RedirectResponse(url="/gradio")


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

    return gr.Dropdown(
        choices=all_records_identifiers,
        interactive=True,
        label="Record Identifier",
        info="Select record to get file list",
    )


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
        # Add explicit embedding dimension and normalization flag
        self.embedding_dim = 768  # MPNet base dimension
        self.normalize_vectors = True
        self.index_type = "flat"  # Document index type used
        self.documents = []
        self.embeddings = None
        self.index = None
        self.max_context_length = 4096
        
    def add_documents(self, new_documents: List[dict]) -> None:
        """Add new documents with validation"""
        for doc in new_documents:
            if not doc.get("content"):
                print(f"Warning: Empty content in document with metadata: {doc.get('metadata', {})}")
                continue
            if len(doc["content"]) < 10:
                print(f"Warning: Very short content ({len(doc['content'])} chars) in document: {doc.get('metadata', {})}")
                continue
            self.documents.append(doc)
        print(f"Added {len(new_documents)} documents. Total documents: {len(self.documents)}")
        
    def validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Validate and normalize embedding shape and values"""
        if embedding.ndim == 3:
            embedding = np.mean(embedding, axis=1)
        embedding = embedding.flatten()
        
        # Add L2 normalization
        if self.normalize_vectors:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
        # Validate dimension
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {embedding.shape[0]}")
            
        return embedding
        
    def build_vector_db(self) -> None:
        """Builds vector database with improved error handling and validation"""
        if not self.documents:
            raise ValueError("No documents to build vector database")
            
        # Process in smaller batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:i + batch_size]
            batch_embeddings = np.zeros((len(batch), self.embedding_dim))
            
            for j, doc in enumerate(batch):
                try:
                    # Get embedding from HF client
                    embedding_response = embeddings_client.post(
                        json={"inputs": doc["content"]},
                        task="feature-extraction"
                    )
                    embedding = np.array(json.loads(embedding_response.decode()))
                    embedding = self.validate_embedding(embedding)
                    batch_embeddings[j] = embedding
                    
                except Exception as e:
                    print(f"Error embedding document {i+j}: {str(e)}")
                    continue
                    
            all_embeddings.append(batch_embeddings)
            
        try:
            self.embeddings = np.vstack(all_embeddings)
            
            # Initialize appropriate FAISS index
            if self.index_type == "flat":
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            elif self.index_type == "ivf":
                # For IVF, we need to train the index
                nlist = min(4096, int(len(self.documents) / 39))
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                # Train on a subset if dataset is large
                train_size = min(100000, len(self.embeddings))
                self.index.train(self.embeddings[:train_size])
                
            self.index.add(self.embeddings)
            print(f"Built index with {len(self.documents)} documents")
            
        except Exception as e:
            print(f"Error building FAISS index: {str(e)}")
            raise

    def search_documents(self, query: str, k: int = 4) -> List[str]:
        """Searches for relevant documents with quality checks"""
        if not self.index:
            print("Warning: Vector database not initialized")
            return ["Vector database not initialized."]
        
        try:
            # Log document count
            print(f"Total documents in index: {len(self.documents)}")
            
            # Get query embedding
            embedding_response = embeddings_client.post(
                json={"inputs": query},
                task="feature-extraction"
            )
            query_embedding = np.array(json.loads(embedding_response.decode()), dtype=np.float32)
            
            query_embedding = self.validate_embedding(query_embedding)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search and log distances
            D, I = self.index.search(query_embedding, k)
            print(f"Search distances: {D[0]}")  # Lower distances = better matches
            
            results_with_metadata = []
            for i, idx in enumerate(I[0]):
                distance = D[0][i]
                similarity_score = 1 / (1 + distance)  # Convert distance to similarity
                
                if similarity_score < 0.7:
                    print(f"Document {i+1} below relevance threshold ({similarity_score:.2f})")
                    continue
                    
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    print(f"\nDocument {i+1} (similarity: {similarity_score:.2f}):")
                    print(f"Record ID: {doc['metadata'].get('record_id', 'unknown')}")
                    print(f"File: {doc['metadata'].get('file_name', 'unknown')}")
                    metadata_str = f"\nSource: Record {doc['metadata'].get('record_id', 'unknown')}, File: {doc['metadata'].get('file_name', 'unknown')}"
                    results_with_metadata.append(doc["content"] + metadata_str)
            
            return results_with_metadata if results_with_metadata else ["No sufficiently relevant documents found."]
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return ["An error occurred during the search."]


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
                
        chunk = text[start:end].strip()
        
        # Log chunk information
        print(f"Chunk size: {len(chunk)} characters")
        if len(chunk) > chunk_size:
            print(f"Warning: Chunk exceeds size limit: {len(chunk)}")
            
        chunks.append(chunk)
        
        # Move the start position forward by the overlap size
        start += chunk_size - overlap_size
        
    # Log overall chunking statistics
    if chunks:
        avg_size = sum(len(c) for c in chunks) / len(chunks)
        print(f"Number of chunks: {len(chunks)}")
        print(f"Average chunk size: {avg_size:.2f} characters")
        
    return chunks


def load_and_chunk_pdf(file_path, chunk_size=1000, overlap=100):
    """Extract and chunk text from PDF with improved handling"""
    try:
        with pymupdf.open(file_path) as pdf:
            full_text = ""
            for page in pdf:
                text = page.get_text()
                if text.strip():  # Only add non-empty pages
                    full_text += text + "\n\n"
            
            if not full_text.strip():
                print(f"Warning: No text extracted from {file_path}")
                return []
            
            # Create chunks with overlap
            chunks = []
            start = 0
            
            while start < len(full_text):
                # Find a good break point
                end = min(start + chunk_size, len(full_text))
                if end < len(full_text):
                    # Try to break at sentence boundary
                    for sep in [". ", ".\n", "\n\n", " "]:
                        break_point = full_text.rfind(sep, start, end)
                        if break_point != -1:
                            end = break_point + 1
                            break
                
                chunk = full_text[start:end].strip()
                if chunk:  # Only add non-empty chunks
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            "start_char": start,
                            "end_char": end,
                            "chunk_size": len(chunk)
                        }
                    })
                
                # Move start position, accounting for overlap
                start = max(start + 1, end - overlap)
            
            print(f"Created {len(chunks)} chunks from PDF")
            return chunks
            
    except Exception as e:
        print(f"Error processing PDF {file_path}: {str(e)}")
        return []


def load_pdf(file_path):
    """Extracts text from a PDF file and stores it in the property documents by page."""

    doc = pymupdf.open(file_path)
    documents = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        documents.append({"page": page_num + 1, "content": text})
    print("PDF processed successfully!")
    return documents


def initialize_rag_system(token, progress=gr.Progress()):
    """Initialize RAG system with improved document processing"""
    progress(0, desc="Starting RAG initialization")
    
    # Create connection to kadi
    manager = KadiManager(instance=instance, host=host, pat=token)
    
    # Get all records with better pagination
    host_api = manager.host if manager.host.endswith("/") else manager.host + "/"
    searched_resource = "records"
    endpoint = urljoin(host_api, searched_resource)
    
    response = manager.search.search_resources("record", per_page=100)
    parsed = json.loads(response.content)
    
    # Get pagination info - handle missing fields
    pagination = parsed.get("_pagination", {})
    total_pages = pagination.get("total_pages", 1)
    per_page = pagination.get("per_page", 100)
    
    # Get initial page items count
    initial_items = len(parsed.get("items", []))
    
    # Estimate total records more safely
    if total_pages == 1:
        total_records = initial_items
    else:
        # Estimate based on full pages plus last page
        total_records = (total_pages - 1) * per_page + initial_items
    
    progress(0.1, desc=f"Found approximately {total_records} records to process")
    
    # Initialize RAG
    rag_system = SimpleRAG()
    processed_records = 0
    
    # Process each record with detailed logging
    for page in range(1, total_pages + 1):
        try:
            progress(0.1 + (0.4 * page/total_pages), 
                    desc=f"Processing records page {page}/{total_pages}")
            
            # Get page data
            if page == 1:
                # Use already fetched first page
                page_data = parsed
            else:
                page_endpoint = endpoint + f"?page={page}&per_page={per_page}"
                response = manager.make_request(page_endpoint)
                page_data = json.loads(response.content)
            
            items = page_data.get("items", [])
            if not items:
                print(f"Warning: No items found on page {page}")
                continue
                
            print(f"Processing page {page} with {len(items)} records")
            
            # Process each record
            for item in items:
                try:
                    record = manager.record(identifier=item["identifier"])
                    file_num = record.get_number_files()
                    print(f"\nProcessing record {item['identifier']} with {file_num} files")
                    
                    if file_num == 0:
                        print(f"Skipping record {item['identifier']}: no files")
                        continue
                    
                    # Calculate pages needed for files
                    pages_needed = (file_num + 99) // 100  # Ceiling division
                    
                    # Get all files in the record
                    for p in range(1, pages_needed + 1):
                        try:
                            files_response = record.get_filelist(page=p, per_page=100)
                            files = files_response.json()["items"]
                            print(f"Processing file page {p}/{pages_needed} ({len(files)} files)")
                            
                            # Process each file
                            for file_info in files:
                                file_id = file_info["id"]
                                file_name = file_info["name"]
                                
                                if file_name.lower().endswith('.pdf'):
                                    print(f"Processing PDF: {file_name}")
                                    with tempfile.TemporaryDirectory(prefix="tmp-kadichat-") as temp_dir:
                                        temp_file_location = os.path.join(temp_dir, file_name)
                                        try:
                                            record.download_file(file_id, temp_file_location)
                                            
                                            # Parse document with chunk size validation
                                            docs = load_and_chunk_pdf(temp_file_location)
                                            print(f"Generated {len(docs)} chunks from {file_name}")
                                            
                                            if not docs:
                                                print(f"Warning: No chunks generated from {file_name}")
                                                continue
                                            
                                            # Add record metadata to each chunk
                                            for doc in docs:
                                                doc["metadata"] = {
                                                    "record_id": item["identifier"],
                                                    "file_name": file_name,
                                                    "chunk_size": len(doc["content"])
                                                }
                                            
                                            # Add chunks to RAG system
                                            rag_system.add_documents(docs)
                                            
                                        except Exception as e:
                                            print(f"Error processing file {file_name}: {str(e)}")
                                            continue
                                            
                        except Exception as e:
                            print(f"Error processing file page {p}: {str(e)}")
                            continue
                    
                    processed_records += 1
                    print(f"Processed {processed_records}/{total_records} records")
                    
                except Exception as e:
                    print(f"Error processing record {item['identifier']}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing page {page}: {str(e)}")
            continue
    
    progress(0.8, desc="Building vector database...")
    
    try:
        if len(rag_system.documents) == 0:
            raise ValueError("No documents were processed successfully")
        
        rag_system.build_vector_db()
        print(f"Successfully built index with {len(rag_system.documents)} documents")
        
        # Test search functionality
        test_results = rag_system.search_documents("test query")
        print(f"Index validation: retrieved {len(test_results)} results")
        
    except Exception as e:
        print(f"Error building vector database: {str(e)}")
        raise
    
    progress(1.0, desc=f"RAG system ready with {len(rag_system.documents)} documents")
    return f"RAG system initialized with {len(rag_system.documents)} documents", rag_system


def preprocess_response(response: str) -> str:
    """Enhanced preprocessing of LLM responses."""
    
    if not response:
        return ""
        
    # Basic cleanup
    response = response.strip()
    
    # Remove common LLM artifacts
    artifacts_to_remove = [
        '<|endoftext|>',
        '<|im_start|>',
        '<|im_end|>',
        '```',
        '===',
    ]
    for artifact in artifacts_to_remove:
        response = response.replace(artifact, '')
    
    # Fix spacing around punctuation
    punctuation_fixes = {
        ' ,': ',',
        ' .': '.',
        ' !': '!',
        ' ?': '?',
        ' :': ':',
        ' ;': ';',
        '( ': '(',
        ' )': ')',
    }
    for wrong, right in punctuation_fixes.items():
        response = response.replace(wrong, right)
    
    # Remove multiple spaces and newlines
    response = ' '.join(response.split())
    
    # Ensure proper sentence structure
    if response and response[0].islower():
        response = response[0].upper() + response[1:]
    
    # Ensure proper ending
    if response and not response[-1] in '.!?':
        response += '.'
    
    return response


@rate_limit(max_per_minute=30)
def respond(message: str, history: List[Tuple[str, str]], user_session_rag) -> Tuple[List[Tuple[str, str]], str]:
    """Enhanced response handler with better context management."""
    
    try:
        if not message.strip():
            return history, "Please provide a valid message."
            
        # Get and log relevant documents
        try:
            retrieved_docs = user_session_rag.search_documents(message)
            print(f"\nRetrieved {len(retrieved_docs)} documents")
        except Exception as e:
            print(f"Document retrieval error: {str(e)}")
            retrieved_docs = []
            
        # Improved context management
        max_context_tokens = 3072  # Reduced from 4096 to leave more room
        max_response_tokens = 1024
        system_message_tokens = 200  # Approximate
        
        available_tokens = max_context_tokens - max_response_tokens - system_message_tokens
        
        # Build context with token counting
        context_parts = []
        total_tokens = 0
        
        for doc in retrieved_docs[:3]:
            doc_tokens = count_tokens(doc)
            print(f"Document tokens: {doc_tokens}")
            
            if total_tokens + doc_tokens > available_tokens:
                print(f"Skipping document, would exceed token limit ({total_tokens + doc_tokens} > {available_tokens})")
                break
                
            context_parts.append(doc)
            total_tokens += doc_tokens
            print(f"Added document. Total tokens: {total_tokens}")
            
        context = "\n\n---\n\n".join(context_parts)
        
        # More focused system message
        system_message = (
            "You are a helpful assistant for answering questions about documents. "
            "Base your response ONLY on the following context documents, separated by '---':\n\n"
            f"{context}\n\n"
            "If the context doesn't contain relevant information, respond with: "
            "'I don't find specific information about that in the available documents.'\n"
            "Keep your response clear, focused and avoid speculation. "
            "Do not use markdown formatting or special characters."
        )
        
        messages = [{"role": "system", "content": system_message}]
        
        # Add limited history
        history_tokens = 0
        recent_history = []
        
        for msg in reversed(history[-3:]):  # Reduced from 5 to 3 messages
            msg_tokens = count_tokens(msg[0]) + count_tokens(msg[1])
            if history_tokens + msg_tokens > available_tokens // 3:  # Reduced history allocation
                break
            recent_history.insert(0, msg)
            history_tokens += msg_tokens
            
        print(f"History tokens: {history_tokens}")
        
        for user_msg, assistant_msg in recent_history:
            messages.extend([
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ])
            
        messages.append({"role": "user", "content": message})
        
        # Log total context size
        total_prompt_tokens = sum(count_tokens(m["content"]) for m in messages)
        print(f"Total prompt tokens: {total_prompt_tokens}")
        
        # Get response with error handling
        try:
            response_content = get_llm_response(messages)
            history.append((message, response_content))
            return history, ""
            
        except ValueError as e:
            return history, f"Response generation error: {str(e)}"
        except Exception as e:
            print(f"Response error: {str(e)}")
            return history, "An error occurred. Please try again."
            
    except Exception as e:
        print(f"Critical error: {str(e)}")
        return history, "An unexpected error occurred."


app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app = gr.mount_gradio_app(app, login_demo, path="/main")

# Gradio interface
with gr.Blocks() as main_demo:
    _state_user_token = gr.State([])
    user_session_rag = gr.State("placeholder")

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
            with gr.Column(scale=3):
                status_box = gr.Textbox(
                    label="Status", 
                    value="Initializing...", 
                    interactive=False
                )
                
        with gr.Row():
            txt_input = gr.Textbox(
                show_label=False,
                placeholder="Type your question here...",
                lines=1
            )
            submit_btn = gr.Button("Submit", scale=1)
            refresh_btn = gr.Button("Refresh Chat", scale=1, variant="secondary")

        # Initialize RAG system on load
        main_demo.load(_init_user_token, None, _state_user_token).then(
            initialize_rag_system,
            inputs=[_state_user_token],
            outputs=[status_box, user_session_rag]
        )

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


def validate_model_tokenizer():
    """Validate LLaMA 3.1 model and tokenizer"""
    try:
        # Test tokenization
        test_text = "Hello, this is a test."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        if not decoded.strip():
            raise ValueError("Tokenizer validation failed")
        
        # Test generation
        outputs = llm_pipeline(
            test_text,
            max_new_tokens=10,
            num_return_sequences=1
        )
        if not outputs[0]["generated_text"].strip():
            raise ValueError("Model validation failed")
        
    except Exception as e:
        print(f"LLaMA 3.1 validation error: {str(e)}")
        raise

# Call validation during startup
validate_model_tokenizer()


if __name__ == "__main__":
    uvicorn.run(app, port=7860, host="0.0.0.0")