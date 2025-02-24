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
        self.documents = []
        self.embeddings_model = None
        self.embeddings = None
        self.index = None
        # Initialize cross-encoder for re-ranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

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
        if self.embeddings_model is None:
            # Using a more powerful embedding model
            self.embeddings_model = SentenceTransformer(
                "hkunlp/instructor-xl",
                trust_remote_code=True
            )

        # Create FAISS index with metadata support
        dimension = 768  # adjust based on model
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add metadata storage
        self.metadata = []
        for doc in self.documents:
            self.metadata.append({
                'file_name': doc.get('metadata', {}).get('file_name', ''),
                'record_id': doc.get('metadata', {}).get('record_id', ''),
                'chunk_id': doc.get('metadata', {}).get('chunk_id', 0)
            })

        # Get embeddings with instruction
        instruction = "Represent the text for retrieving relevant scientific document passages:"
        texts = [instruction + doc["content"] for doc in self.documents]
        self.embeddings = self.embeddings_model.encode(texts, show_progress_bar=True)
        
        self.index.add(np.array(self.embeddings))

    def search_documents(self, query: str, k: int = 8, threshold: float = 1000.0) -> List[str]:
        """Enhanced search with re-ranking and hybrid retrieval."""
        print(f"Searching for query: '{query[:50]}...' with k={k}")
        
        # Hybrid search preparation
        query_terms = set(query.lower().split())
        
        # Get initial candidates through vector similarity
        instruction = "Represent the question for retrieving relevant scientific document passages:"
        query_embedding = self.embeddings_model.encode([instruction + query])
        D, I = self.index.search(query_embedding, k * 2)  # Get more candidates for re-ranking
        
        print(f"Found {len(I[0])} initial candidates")
        
        candidates = []
        for distance, idx in zip(D[0], I[0]):
            if distance >= threshold:
                continue
                
            doc = self.documents[idx]
            content = doc["content"]
            metadata = self.metadata[idx]
            
            # Hybrid scoring
            term_overlap = len(query_terms.intersection(set(content.lower().split())))
            hybrid_score = distance * (1.0 - (term_overlap * 0.1))
            
            candidates.append({
                'content': content,
                'distance': distance,
                'hybrid_score': hybrid_score,
                'metadata': metadata
            })
        
        # Re-rank using cross-encoder
        if candidates:
            pairs = [(query, doc['content']) for doc in candidates]
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Combine scores
            for idx, doc in enumerate(candidates):
                doc['final_score'] = (doc['hybrid_score'] + cross_scores[idx]) / 2
            
            # Sort and select top k
            candidates.sort(key=lambda x: x['final_score'])
            candidates = candidates[:k]
        
        # Format results
        results = []
        for doc in candidates:
            metadata = doc['metadata']
            context = f"[Source: {metadata['file_name']}, Record: {metadata['record_id']}, "
            context += f"Relevance: {1 - doc['final_score']/threshold:.2f}]"
            results.append(f"{context}\n{doc['content']}")
        
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


def process_file(file_path):
    """Process a file based on its extension."""
    file_extension = file_path.lower().split('.')[-1]
    
    processors = {
        'pdf': load_and_chunk_pdf,
        'txt': load_text_file,
        'md': load_markdown_file,
        'docx': load_docx
    }
    
    processor = processors.get(file_extension)
    if processor is None:
        print(f"Unsupported file type: {file_extension}")
        return []
        
    try:
        return processor(file_path)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []


def prepare_file_for_chat(record_id, file_names, token, progress=gr.Progress()):
    """Parse file and prepare RAG."""
    
    progress(0, desc="Starting")
    # Create connection to kadi
    manager = KadiManager(instance=instance, host=host, pat=token)
    record = manager.record(identifier=record_id)
    
    progress(0.2, desc="Loading files...")
    # Get all files in record
    file_list = record.get_filelist().json()["items"]
    
    # Filter files by extension
    allowed_extensions = ['.docx', '.pdf', '.md', '.txt']
    valid_files = [
        file for file in file_list 
        if any(file["name"].lower().endswith(ext) for ext in allowed_extensions)
    ]
    
    if not valid_files:
        raise gr.Error("No valid files found in record")
        
    # Parse files
    documents = []
    progress(0.3, desc=f"Processing {len(valid_files)} files...")
    
    # Download and process each file
    for file_info in valid_files:
        file_name = file_info["name"]
        file_id = file_info["id"]
        
        with tempfile.TemporaryDirectory(prefix="tmp-kadichat-downloads-") as temp_dir:
            temp_file_location = os.path.join(temp_dir, file_name)
            record.download_file(file_id, temp_file_location)
            
            try:
                # Handle different file types
                if file_name.lower().endswith('.pdf'):
                    docs = load_and_chunk_pdf(temp_file_location)
                else:
                    # For text-based files, read directly
                    with open(temp_file_location, 'r', encoding='utf-8') as f:
                        text = f.read()
                        docs = [{"content": chunk, "metadata": {"filename": file_name}} 
                               for chunk in chunk_text(text)]
                documents.extend(docs)
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue

    progress(0.6, desc="Embedding documents...")
    user_rag = SimpleRAG()
    user_rag.documents = documents
    user_rag.embeddings_model = embeddings_model
    user_rag.build_vector_db()
    
    progress(1, desc=f"Ready to chat - processed {len(valid_files)} files")
    return f"Ready to chat - processed {len(valid_files)} files", user_rag


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
    """Enhanced response generation with simpler prompting."""
    # Get relevant documents
    retrieved_docs = user_session_rag.search_documents(message, k=4)
    context = "\n\n".join(retrieved_docs)
    
    # Simple, focused prompt
    system_message = """You are a helpful assistant that answers questions based on the provided documents.
    Use only the information from the documents to answer questions.
    If you cannot find relevant information, say so clearly.
    
    Context:
    {}""".format(context)
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": message}
    ]
    
    # Get response with zero temperature for consistency
    response = client.chat_completion(
        messages,
        max_tokens=2048,
        temperature=0.0
    )
    
    response_content = "".join([
        choice.message["content"]
        for choice in response.choices
        if "content" in choice.message
    ])
    
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
                record_list = gr.Dropdown(label="Record Identifier")
                
                gr.Markdown("  " * 200)
                main_demo.load(_init_user_token, None, _state_user_token).then(
                    get_all_records, _state_user_token, record_list
                )

                parse_files = gr.Button("Process Record Files")
                message_box = gr.Textbox(
                    label="", value="progress bar", interactive=False
                )
                
                # Create a hidden textbox for file_names parameter
                file_names = gr.Textbox(visible=False, value="[]")
                
                # Modified click handler
                parse_files.click(
                    fn=prepare_file_for_chat,
                    inputs=[record_list, file_names, _state_user_token],
                    outputs=[message_box, user_session_rag],
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