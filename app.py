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
from transformers import AutoTokenizer
import time
from functools import wraps

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

client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")

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
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

def count_tokens(text):
    """Count tokens in text using the model's tokenizer."""
    return len(tokenizer.encode(text))

def validate_response(response_content: str) -> Tuple[bool, str]:
    """Validate response quality and return reason if invalid."""
    
    # Check for empty or very short responses
    if not response_content or len(response_content.strip()) < 10:
        return False, "Response too short"
        
    # Check for repetitive patterns
    words = response_content.split()
    if len(words) > 3:
        repeated_words = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
        if repeated_words > len(words) * 0.3:  # More than 30% repeated
            return False, "Too many repeated words"
            
    # Check for corruption markers - refined to be more specific
    corruption_markers = [
        "the the",
        "protein the",
        "<|",
        "|>",
        "<<",
        ">>",
        "```",
        "{{",
        "}}",
        "\x00",  # null byte
        "\u0000"  # unicode null
    ]
    
    # Check for numbered list at start of line
    lines = response_content.split('\n')
    for line in lines:
        if line.strip().startswith('1.') and len(line.strip()) <= 3:
            return False, "Contains bare numbered list marker"
            
    for marker in corruption_markers:
        if marker in response_content:
            return False, f"Contains corruption marker: {marker}"
            
    # Check for coherent sentence structure
    if not any(c.isupper() for c in response_content[:10]):
        return False, "No capitalization in first 10 characters"
        
    # More lenient ending punctuation check
    if not response_content.strip()[-1] in '.!?..."\'':
        return False, "Missing ending punctuation"
        
    return True, "Valid response"

@rate_limit(max_per_minute=30)
def get_llm_response(messages):
    """Get response from LLM with rate limiting."""
    return client.chat_completion(
        messages,
        max_tokens=1024,
        temperature=0.0
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
        self.documents = []
        self.embeddings_model = None
        self.embeddings = None
        self.index = None
        self.max_context_length = 4096
        self.embedding_dim = 768  # Base embedding dimension for the model
        
    def add_documents(self, new_documents: List[dict]) -> None:
        """Add new documents to the existing collection"""
        self.documents.extend(new_documents)
        
    def validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Validate and normalize embedding shape"""
        # Handle 3D embeddings by taking mean across sequence length
        if embedding.ndim == 3:
            embedding = np.mean(embedding, axis=1)
        return embedding.flatten()
        
    def build_vector_db(self) -> None:
        """Builds a vector database using all documents"""
        if not self.documents:
            print("No documents to build vector database")
            return
            
        contents = [doc["content"] for doc in self.documents]
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            batch_embeddings = np.zeros((len(batch), self.embedding_dim), dtype=np.float32)
            
            for j, text in enumerate(batch):
                try:
                    embedding_response = embeddings_client.post(
                        json={"inputs": text},
                        task="feature-extraction"
                    )
                    embedding = np.array(json.loads(embedding_response.decode()), dtype=np.float32)
                    embedding = self.validate_embedding(embedding)
                    
                    # Ensure correct dimension
                    if embedding.shape[0] != self.embedding_dim:
                        print(f"Warning: Inconsistent embedding dimension for document {i+j}")
                        continue
                        
                    batch_embeddings[j] = embedding
                    
                except Exception as e:
                    print(f"Error processing document {i+j}: {str(e)}")
                    continue
            
            all_embeddings.append(batch_embeddings)
            print(f"Processed batch {i//batch_size + 1}/{(len(contents) + batch_size - 1)//batch_size}")
        
        try:
            self.embeddings = np.vstack(all_embeddings)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(self.embeddings)
            print(f"Vector database built successfully with {len(self.documents)} documents!")
            
        except Exception as e:
            print(f"Error building index: {str(e)}")
            raise

    def search_documents(self, query: str, k: int = 4) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        if not self.index:
            return ["Vector database not initialized."]
            
        try:
            # Get query embedding
            embedding_response = embeddings_client.post(
                json={"inputs": query},
                task="feature-extraction"
            )
            # Convert response to embedding vector
            query_embedding = np.array(json.loads(embedding_response.decode()), dtype=np.float32)
            print(f"Raw query embedding shape: {query_embedding.shape}")
            
            query_embedding = self.validate_embedding(query_embedding)
            print(f"Flattened query embedding shape: {query_embedding.shape}")
            
            # Ensure correct shape for FAISS
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search similar documents
            D, I = self.index.search(query_embedding, k)
            
            # Add metadata to results
            results_with_metadata = []
            for idx in I[0]:
                if idx < len(self.documents):  # Guard against index out of bounds
                    doc = self.documents[idx]
                    metadata_str = f"\nSource: Record {doc['metadata'].get('record_id', 'unknown')}, File: {doc['metadata'].get('file_name', 'unknown')}"
                    results_with_metadata.append(doc["content"] + metadata_str)
            
            return results_with_metadata if results_with_metadata else ["No relevant documents found."]
            
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
    """Initialize RAG system with all records from Kadi"""
    progress(0, desc="Starting RAG initialization")
    
    # Create connection to kadi
    manager = KadiManager(instance=instance, host=host, pat=token)
    
    # Get all records
    host_api = manager.host if manager.host.endswith("/") else manager.host + "/"
    searched_resource = "records"
    endpoint = urljoin(host_api, searched_resource)
    
    response = manager.search.search_resources("record", per_page=100)
    parsed = json.loads(response.content)
    total_pages = parsed["_pagination"]["total_pages"]
    
    progress(0.1, desc="Collecting records...")
    
    # Initialize RAG
    rag_system = SimpleRAG()
    
    # Process each record
    for page in range(1, total_pages + 1):
        progress(0.1 + (0.4 * page/total_pages), desc=f"Processing records page {page}/{total_pages}")
        
        page_endpoint = endpoint + f"?page={page}&per_page=100"
        response = manager.make_request(page_endpoint)
        parsed = json.loads(response.content)
        
        for item in parsed["items"]:
            record = manager.record(identifier=item["identifier"])
            file_num = record.get_number_files()
            
            # Get all files in the record
            for p in range(1, (file_num // 100) + 2):
                files = record.get_filelist(page=p, per_page=100).json()["items"]
                
                # Process each file
                for file_info in files:
                    file_id = file_info["id"]
                    file_name = file_info["name"]
                    
                    if file_name.lower().endswith('.pdf'):
                        with tempfile.TemporaryDirectory(prefix="tmp-kadichat-") as temp_dir:
                            temp_file_location = os.path.join(temp_dir, file_name)
                            record.download_file(file_id, temp_file_location)
                            
                            # Parse document and add to RAG
                            docs = load_and_chunk_pdf(temp_file_location)
                            # Add record metadata to each chunk
                            for doc in docs:
                                doc["metadata"].update({
                                    "record_id": item["identifier"],
                                    "file_name": file_name
                                })
                            rag_system.add_documents(docs)
    
    progress(0.8, desc="Building vector database...")
    rag_system.build_vector_db()
    
    progress(1.0, desc="RAG system ready")
    return "RAG system initialized and ready", rag_system


def preprocess_response(response: str) -> str:
    """Preprocesses the response to make it more polished."""
    
    # Basic cleanup
    response = response.strip()
    
    # Fix common formatting issues
    response = response.replace(" ,", ",")
    response = response.replace(" .", ".")
    response = response.replace(" !", "!")
    response = response.replace(" ?", "?")
    
    # Remove multiple spaces
    response = " ".join(response.split())
    
    # Ensure proper sentence structure
    if not response[0].isupper():
        response = response[0].upper() + response[1:]
    if not response[-1] in '.!?':
        response += '.'
        
    # Remove any remaining markdown or code formatting
    response = response.replace('```', '')
    response = response.replace('`', '')
    
    return response


@rate_limit(max_per_minute=30)
def respond(message: str, history: List[Tuple[str, str]], user_session_rag):
    """Get response from LLMs with improved error handling."""
    try:
        # Limit history size
        max_history_items = 3
        recent_history = history[-max_history_items:] if history else []
        
        # Get relevant documents
        retrieved_docs = user_session_rag.search_documents(message)
        
        # Manage context size
        max_context_chars = 6000
        context = ""
        for doc in retrieved_docs:
            if len(context) + len(doc) < max_context_chars:
                context += doc + "\n"
            else:
                break
                
        # Structure prompt clearly
        system_message = (
            "You are an assistant helping users with questions about Kadi. "
            "Base your response on these relevant documents:\n\n"
            f"{context}\n\n"
            "If you cannot find relevant information in the documents, "
            "say so clearly instead of making things up."
        )
        
        # Check total tokens
        total_tokens = count_tokens(system_message + message)
        if total_tokens > 4000:  # Adjust based on model limits
            print(f"Warning: Input too long ({total_tokens} tokens)")
            context = context[:int(len(context)/2)]
            system_message = system_message.replace(context, context[:int(len(context)/2)])
            
        # Build messages with limited history
        messages = [{"role": "system", "content": system_message}]
        for user_msg, assistant_msg in recent_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})
        
        # Get response with retries and better error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = get_llm_response(messages)
                response_content = "".join([
                    choice.message["content"]
                    for choice in response.choices
                    if "content" in choice.message
                ])
                
                # Enhanced validation
                is_valid, reason = validate_response(response_content)
                if not is_valid:
                    print(f"Response validation failed: {reason}")
                    if attempt == max_retries - 1:
                        return history, f"I apologize, but I'm having trouble generating a proper response ({reason}). Please try again."
                    continue
                    
                # Process valid response
                response_content = preprocess_response(response_content)
                history.append((message, response_content))
                return history, ""
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return history, "I apologize, but I'm having technical difficulties. Please try again in a moment."
                time.sleep(1)  # Wait before retry
                
    except Exception as e:
        print(f"Error in respond function: {str(e)}")
        return history, "An unexpected error occurred. Please try again."


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


if __name__ == "__main__":
    uvicorn.run(app, port=7860, host="0.0.0.0")