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


def get_files_in_record(record_list, user_token):
    """Get all files with specific extensions from records."""
    
    if not record_list:
        return []
        
    manager = KadiManager(instance=instance, host=host, pat=user_token)
    allowed_extensions = ['.docx', '.pdf', '.md', '.txt']
    file_list = []

    for record_id in record_list:
        try:
            record = manager.record(identifier=record_id)
            response = record.get_filelist()
            if response.ok:
                files = response.json()['items']
                for file in files:
                    file_name = file['name']
                    file_ext = os.path.splitext(file_name)[1].lower()
                    if file_ext in allowed_extensions:
                        file_list.append({
                            'record_id': record_id,
                            'file_name': file_name,
                            'file_id': file['id']
                        })
        except kadi_apy.lib.exceptions.KadiAPYRequestError as e:
            print(f"Error processing record {record_id}: {e}")
            continue

    return file_list


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

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        if self.embeddings_model is None:
            self.embeddings_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2", 
                trust_remote_code=True
            )

        # Get embeddings for document contents
        contents = [doc["content"] for doc in self.documents]
        self.embeddings = self.embeddings_model.encode(
            contents,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings))

    def search_documents(self, query: str, k: int = 4) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        if not self.documents:
            return ["No documents available."]
            
        print(f"Searching for query: '{query}' with k={k}")
        
        # Get query embedding
        query_embedding = self.embeddings_model.encode([query])
        
        # Search index
        D, I = self.index.search(np.array(query_embedding), k)
        print(f"Found {len(I[0])} matches")
        
        # Return content of matched documents
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx]["content"])
                
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


def prepare_file_for_chat(record_list, token, progress=gr.Progress()):
    """Parse files and prepare RAG."""
    
    progress(0, desc="Starting")
    manager = KadiManager(instance=instance, host=host, pat=token)
    
    # Get all valid files
    progress(0.2, desc="Finding valid files...")
    file_list = get_files_in_record(record_list, token)
    
    if not file_list:
        raise gr.Error("No valid files found")
    
    # Parse files
    progress(0.3, desc="Loading files...")
    documents = []
    
    with tempfile.TemporaryDirectory(prefix="tmp-kadichat-downloads-") as temp_dir:
        for file_info in file_list:
            try:
                record = manager.record(identifier=file_info['record_id'])
                temp_file_location = os.path.join(temp_dir, file_info['file_name'])
                
                # Download file
                record.download_file(file_info['file_id'], temp_file_location)
                
                # Parse document based on extension
                file_ext = os.path.splitext(file_info['file_name'])[1].lower()
                if file_ext == '.pdf':
                    docs = load_and_chunk_pdf(temp_file_location)
                else:  # For .txt, .md, .docx
                    with open(temp_file_location, 'r', encoding='utf-8') as f:
                        text = f.read()
                        docs = [{'content': chunk} for chunk in chunk_text(text)]
                
                documents.extend(docs)
                
            except Exception as e:
                print(f"Error processing file {file_info['file_name']}: {e}")
                continue

    progress(0.6, desc="Embedding documents...")
    user_rag = SimpleRAG()
    user_rag.documents = documents
    user_rag.embeddings_model = embeddings_model
    user_rag.build_vector_db()
    
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
    """Get respond from LLMs."""
    # Get relevant documents
    retrieved_docs = user_session_rag.search_documents(message, k=4)
    
    # Combine retrieved documents into context
    context = "\n---\n".join(retrieved_docs)
    
    # Create system message with context
    system_message = """You are an assistant to help users answer questions related to Kadi and research papers. 
    Use the following relevant documents to answer the question. If you cannot find relevant information in the documents, say so.
    
    Relevant documents:
    {}""".format(context)
    
    messages = [
        {"role": "assistant", "content": system_message},
        {"role": "user", "content": message}
    ]

    # Get response from LLM
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
                    fn=prepare_file_for_chat,
                    inputs=[record_list, _state_user_token],
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