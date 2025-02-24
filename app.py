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
from sentence_transformers import CrossEncoder
from docx import Document
import markdown

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


def get_files_in_record(all_records_identifiers, user_token):
    """Get all file lists from all records."""
    print(f"Starting get_files_in_record with {len(all_records_identifiers)} records")  # Debug print
    
    if not all_records_identifiers:
        print("No records provided")  # Debug print
        return []
        
    manager = KadiManager(instance=instance, host=host, pat=user_token)
    all_file_names = []

    for record_id in all_records_identifiers:
        try:
            record = manager.record(identifier=record_id)
            file_num = record.get_number_files()

            per_page = 100  # default in kadi
            not_divisible = file_num % per_page
            page_num = (file_num // per_page + 1) if not_divisible else (file_num // per_page)

            for p in range(1, page_num + 1):  # page starts at 1 in kadi
                file_names = [
                    info["name"]
                    for info in record.get_filelist(page=p, per_page=per_page).json()["items"]
                ]
                all_file_names.extend(file_names)

        except kadi_apy.lib.exceptions.KadiAPYInputError as e:
            print(f"Error accessing record {record_id}: {e}")
            continue

    return all_file_names

def get_all_records(user_token, progress=gr.Progress()):
    """Get all record list in Kadi."""
    print("Starting get_all_records with token:", user_token)  # Debug print
    progress(0, desc="Starting record collection...")
    if not user_token:
        print("No token provided")  # Debug print
        return []

    try:
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

        progress(0.5, desc="Fetching records...")
        all_records_identifiers = []
        for page in range(1, total_pages + 1):
            progress(0.5 + (0.5 * page/total_pages), desc=f"Processing page {page}/{total_pages}")
            page_endpoint = endpoint + f"?page={page}&per_page=100"
            response = manager.make_request(page_endpoint)
            parsed = json.loads(response.content)
            all_records_identifiers.extend(get_page_records(parsed))

        print(f"Found {len(all_records_identifiers)} records")  # Debug print
        return all_records_identifiers
    except Exception as e:
        print(f"Error in get_all_records: {e}")  # Debug print
        return []


def _init_user_token(request: gr.Request):
    """Init user token."""
    try:
        user_token = request.request.session["user_access_token"]
        debug_msg = f"Token initialized: {user_token is not None}"
        print(debug_msg)  # Debug print
        return user_token, debug_msg
    except Exception as e:
        error_msg = f"Error initializing token: {e}"
        print(error_msg)
        return None, error_msg


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
        print("\n=== Initializing SimpleRAG ===")
        self.documents = []
        self.embeddings_model = None
        self.embeddings = None
        self.index = None
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("SimpleRAG initialized successfully")

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the property documents by page."""

        doc = pymupdf.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        # print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database with improved indexing."""
        print("\n=== Building Vector Database ===")
        print(f"Number of documents to process: {len(self.documents)}")
        
        if self.embeddings_model is None:
            print("Initializing embedding model: hkunlp/instructor-xl")
            self.embeddings_model = SentenceTransformer(
                "hkunlp/instructor-xl",
                trust_remote_code=True
            )

        # Create FAISS index
        dimension = 768
        print(f"Creating FAISS index with dimension: {dimension}")
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add metadata storage
        print("Processing document metadata")
        self.metadata = []
        for doc in self.documents:
            self.metadata.append({
                'file_name': doc.get('metadata', {}).get('file_name', ''),
                'record_id': doc.get('metadata', {}).get('record_id', ''),
                'chunk_id': doc.get('metadata', {}).get('chunk_id', 0)
            })

        # Get embeddings
        print("Generating embeddings...")
        instruction = "Represent the text for retrieving relevant scientific document passages:"
        texts = [instruction + doc["content"] for doc in self.documents]
        self.embeddings = self.embeddings_model.encode(texts, show_progress_bar=True)
        
        print(f"Adding {len(self.embeddings)} embeddings to index")
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully")

    def search_documents(self, query: str, k: int = 8, threshold: float = 1000.0) -> List[str]:
        """Enhanced search with re-ranking and hybrid retrieval."""
        print(f"\n=== Searching Documents for Query: {query} ===")
        
        # Hybrid search preparation
        query_terms = set(query.lower().split())
        print(f"Query terms: {query_terms}")
        
        # Get initial candidates
        print("Generating query embedding...")
        instruction = "Represent the question for retrieving relevant scientific document passages:"
        query_embedding = self.embeddings_model.encode([instruction + query])
        
        print(f"Searching index for top {k*2} candidates...")
        D, I = self.index.search(query_embedding, k * 2)
        
        candidates = []
        print("\n=== Processing Search Results ===")
        for distance, idx in zip(D[0], I[0]):
            if distance >= threshold:
                print(f"Skipping document {idx} (distance {distance:.2f} >= threshold {threshold})")
                continue
                
            doc = self.documents[idx]
            content = doc["content"]
            metadata = self.metadata[idx]
            
            # Hybrid scoring
            term_overlap = len(query_terms.intersection(set(content.lower().split())))
            hybrid_score = distance * (1.0 - (term_overlap * 0.1))
            
            print(f"\nDocument {idx}:")
            print(f"Distance: {distance:.2f}")
            print(f"Term overlap: {term_overlap}")
            print(f"Hybrid score: {hybrid_score:.2f}")
            
            candidates.append({
                'content': content,
                'distance': distance,
                'hybrid_score': hybrid_score,
                'metadata': metadata
            })
        
        # Re-rank using cross-encoder
        if candidates:
            print("\n=== Re-ranking Candidates ===")
            pairs = [(query, doc['content']) for doc in candidates]
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Combine scores
            for idx, doc in enumerate(candidates):
                doc['final_score'] = (doc['hybrid_score'] + cross_scores[idx]) / 2
                print(f"Document {idx} final score: {doc['final_score']:.2f}")
            
            # Sort and select top k
            candidates.sort(key=lambda x: x['final_score'])
            candidates = candidates[:k]
        
        # Format results
        results = []
        print("\n=== Final Results ===")
        for i, doc in enumerate(candidates):
            metadata = doc['metadata']
            context = f"[Source: {metadata['file_name']}, Record: {metadata['record_id']}, "
            context += f"Relevance: {1 - doc['final_score']/threshold:.2f}]"
            results.append(f"{context}\n{doc['content']}")
            print(f"Result {i+1} metadata: {context}")
        
        print(f"Returning {len(results)} results")
        return results if results else ["No relevant documents found."]


def chunk_text(text, chunk_size=2048, overlap_size=256, separators=["\n\n", "\n"]):
    """Chunk text into pieces of specified size with overlap, considering separators."""
    print(f"\nDEBUG: Chunking text of length {len(text)} characters")
    print(f"DEBUG: Using chunk_size={chunk_size}, overlap_size={overlap_size}")

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
        
        print(f"DEBUG: Created chunk {len(chunks)} with length {len(chunk)} characters")

        # Move the start position forward by the overlap size
        start += chunk_size - overlap_size

    print(f"DEBUG: Final number of chunks: {len(chunks)}")
    return chunks


def load_and_chunk_pdf(file_path):
    """Extracts text from a PDF file and stores it in the property documents by chunks."""
    print(f"\nDEBUG: Processing PDF: {file_path}")

    with pymupdf.open(file_path) as pdf:
        text = ""
        for page_num, page in enumerate(pdf):
            page_text = page.get_text()
            text += page_text
            print(f"DEBUG: Page {page_num + 1} length: {len(page_text)} characters")

        chunks = chunk_text(text)
        print(f"DEBUG: Created {len(chunks)} chunks from PDF")
        print(f"DEBUG: Average chunk size: {sum(len(c) for c in chunks)/len(chunks):.0f} characters")
        print(f"DEBUG: First chunk preview: {chunks[0][:200]}...")
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "content": chunk,
                "metadata": {**pdf.metadata, "chunk_id": i}
            })

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


def load_text_file(file_path):
    """Extracts text from a plain text file."""
    print(f"\nDEBUG: Processing text file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
    chunks = chunk_text(text)
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "content": chunk,
            "metadata": {"file_type": "text", "chunk_id": i}
        })
    
    return documents


def load_markdown_file(file_path):
    """Extracts text from a markdown file."""
    print(f"\nDEBUG: Processing markdown file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        md_text = file.read()
        # Convert markdown to plain text
        html = markdown.markdown(md_text)
        # Simple HTML tag removal (you might want to use a proper HTML parser for better results)
        text = html.replace('<p>', '\n\n').replace('</p>', '').replace('<br>', '\n')
        
    chunks = chunk_text(text)
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "content": chunk,
            "metadata": {"file_type": "markdown", "chunk_id": i}
        })
    
    return documents


def load_docx(file_path):
    """Extracts text from a Word document."""
    print(f"\nDEBUG: Processing Word document: {file_path}")
    
    doc = Document(file_path)
    text = '\n\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    chunks = chunk_text(text)
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "content": chunk,
            "metadata": {"file_type": "docx", "chunk_id": i}
        })
    
    return documents


def process_file(file_path: str) -> List[str]:
    """Process a single document file and generate chunks."""
    file_type = file_path.split('.')[-1].lower()
    chunks = []
    
    print(f"\nProcessing file: {file_path} of type {file_type}")
    
    try:
        if file_type == "pdf":
            doc = pymupdf.open(file_path)
            text = ""
            print(f"PDF pages: {doc.page_count}")
            for page in doc:
                page_text = page.get_text()
                print(f"Page text length: {len(page_text)}")
                text += page_text
                
        elif file_type == "docx":
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            print(f"DOCX text length: {len(text)}")
            
        elif file_type in ["txt", "md"]:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                print(f"{file_type.upper()} text length: {len(text)}")
                
        else:
            print(f"Skipping unsupported file type: {file_type}")
            return []

        # Only process if we have meaningful text
        if len(text.strip()) > 0:
            chunks = create_chunks(text)
            print(f"Generated {len(chunks)} chunks from text of length {len(text)}")
        else:
            print("No text content found in file")
            
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return []
        
    return chunks

def create_chunks(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Create overlapping chunks from text."""
    chunks = []
    start = 0
    text_length = len(text)
    
    print(f"Creating chunks from text of length {text_length}")
    print(f"Using max_chunk_size={max_chunk_size}, overlap={overlap}")
    
    while start < text_length:
        end = start + max_chunk_size
        if end > text_length:
            end = text_length
        
        # Adjust end to avoid breaking sentences
        if end < text_length:
            # Look for sentence boundaries (., !, ?)
            for i in range(end, start, -1):
                if text[i-1] in '.!?' and (i == text_length or text[i].isspace()):
                    end = i
                    break
                    
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
            print(f"Created chunk of length {len(chunk)}")
            
        start = end - overlap
        
    return chunks


def prepare_file_for_chat(all_records_identifiers, all_file_names, token, progress=gr.Progress()):
    """Parse file and prepare RAG."""
    print("\n=== Starting prepare_file_for_chat ===")
    print(f"Number of records: {len(all_records_identifiers)}")
    print(f"Number of files: {len(all_file_names)}")

    if not all_file_names:
        print("ERROR: No files found")
        raise gr.Error("No files found")
    progress(0, desc="Starting")
    
    # Create connection to kadi
    manager = KadiManager(instance=instance, host=host, pat=token)
    documents = []
    
    total_files = len(all_file_names)
    files_processed = 0
    
    print("\n=== Processing Records and Files ===")
    # Iterate through all records
    for record_id in all_records_identifiers:
        try:
            print(f"\nProcessing record: {record_id}")
            record = manager.record(identifier=record_id)
            
            # Get all files for this record
            record_files = record.get_filelist().json()["items"]
            print(f"Found {len(record_files)} files in record")
            
            # Process each file in this record
            for file_info in record_files:
                file_name = file_info["name"]
                if file_name in all_file_names:  # Only process files we want
                    print(f"\nProcessing file: {file_name}")
                    progress(0.2 + (0.6 * files_processed/total_files), 
                            desc=f"Processing {file_name}...")
                    
                    file_id = file_info["id"]
                    with tempfile.TemporaryDirectory(prefix="tmp-kadichat-downloads-") as temp_dir:
                        temp_file_location = os.path.join(temp_dir, file_name)
                        record.download_file(file_id, temp_file_location)
                        print(f"Downloaded file to: {temp_file_location}")
                        
                        # Parse document
                        try:
                            docs = process_file(temp_file_location)
                            print(f"Successfully processed file. Generated {len(docs)} chunks")
                            # Add source information to each chunk
                            for doc in docs:
                                doc["metadata"].update({
                                    "file_name": file_name,
                                    "record_id": record_id
                                })
                            documents.extend(docs)
                        except Exception as e:
                            print(f"ERROR processing file {file_name}: {str(e)}")
                            continue
                            
                    files_processed += 1
                    print(f"Total files processed: {files_processed}/{total_files}")
                    
        except kadi_apy.lib.exceptions.KadiAPYInputError as e:
            print(f"ERROR accessing record {record_id}: {str(e)}")
            continue

    print(f"\n=== Document Processing Summary ===")
    print(f"Total documents processed: {len(documents)}")
    if not documents:
        print("ERROR: No documents were successfully processed")
        raise gr.Error("No documents were successfully processed")

    print("\n=== Building Vector Database ===")
    progress(0.8, desc="Building vector database...")
    user_rag = SimpleRAG()
    user_rag.documents = documents
    user_rag.embeddings_model = embeddings_model
    user_rag.build_vector_db()
    
    print(f"Vector database built successfully with {len(documents)} documents")
    progress(1, desc="Ready to chat")
    return "Ready to chat", user_rag


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
    """Enhanced response generation with better prompting."""
    print("\n=== Starting New Response Generation ===")
    print(f"User message: {message}")
    
    # Get relevant documents
    print("\n=== Retrieving Relevant Documents ===")
    retrieved_docs = user_session_rag.search_documents(message)
    print(f"Retrieved {len(retrieved_docs)} documents")
    
    # Print detailed information about retrieved documents
    for i, doc in enumerate(retrieved_docs):
        print(f"\nDocument {i+1}:")
        # Extract metadata if present
        metadata_start = doc.find("[Source:")
        metadata_end = doc.find("]", metadata_start) + 1 if metadata_start != -1 else 0
        metadata = doc[metadata_start:metadata_end] if metadata_start != -1 else "No metadata"
        print(f"Metadata: {metadata}")
        print(f"Content preview: {doc[metadata_end:metadata_end+200]}...")
    
    context = "\n".join(retrieved_docs)
    print(f"\nTotal context length: {len(context)} characters")
    
    # Enhanced system message with stronger instruction
    system_message = """You are an expert assistant. You MUST use the provided context to answer questions.
    If you cannot find relevant information in the context, say so explicitly.
    
    Context:
    {}
    
    Use the above context to answer the following question. If you cannot find relevant information
    in the context, acknowledge this fact.""".format(context)

    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": message})
    
    print("\n=== Generating Response ===")
    response = client.chat_completion(
        messages,
        max_tokens=2048,
        temperature=0.1,
    )
    
    response_content = "".join([
        choice.message["content"]
        for choice in response.choices
        if "content" in choice.message
    ])
    
    print("\n=== Response Summary ===")
    print(f"Response length: {len(response_content)} characters")
    print(f"Response preview: {response_content[:200]}...")
    
    history.append((message, response_content))
    return history, ""


app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app = gr.mount_gradio_app(app, login_demo, path="/main")

# Gradio interface
with gr.Blocks() as main_demo:

    # State for storing user token
    _state_user_token = gr.State([])

    # State for user rag
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
                record_list = gr.State([])
                file_list = gr.State([])
                
                load_files_btn = gr.Button("Load All Files")
                progress_box = gr.Textbox(label="Progress", value="Click 'Load All Files' to start", interactive=False)
                debug_box = gr.Textbox(label="Debug Info", interactive=False)
                
                # Initialize user token with debug output
                main_demo.load(
                    _init_user_token, 
                    None, 
                    [_state_user_token, debug_box],
                )

                def debug_step(step_name, data):
                    print(f"Debug {step_name}:", data)
                    return data, f"Completed {step_name} with data length: {len(data) if isinstance(data, list) else 'N/A'}"

                # Chain of operations when button is clicked
                load_files_btn.click(
                    fn=lambda x: (x, f"Starting chain with token: {x}"),
                    inputs=[_state_user_token],
                    outputs=[_state_user_token, debug_box],
                ).success(
                    fn=get_all_records,
                    inputs=[_state_user_token],
                    outputs=[record_list],
                    show_progress=True,
                ).success(
                    fn=lambda x: debug_step("get_all_records", x),
                    inputs=[record_list],
                    outputs=[record_list, debug_box],
                ).success(
                    fn=get_files_in_record,
                    inputs=[record_list, _state_user_token],
                    outputs=[file_list],
                    show_progress=True,
                ).success(
                    fn=lambda x: debug_step("get_files_in_record", x),
                    inputs=[file_list],
                    outputs=[file_list, debug_box],
                ).success(
                    fn=prepare_file_for_chat,
                    inputs=[record_list, file_list, _state_user_token],
                    outputs=[progress_box, user_session_rag],
                    show_progress=True,
                )



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