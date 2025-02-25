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


def get_files_in_record(all_records_identifiers, user_token, progress=gr.Progress()):
    """Get all file list within one record."""
    if not all_records_identifiers:
        return [], "No records found", {}

    progress(0, desc="Connecting to Kadi...")
    manager = KadiManager(instance=instance, host=host, pat=user_token)

    file_names = []
    file_record_mapping = {}
    total_records = len(all_records_identifiers)
    
    for idx, record_id in enumerate(all_records_identifiers):
        progress_val = idx / total_records
        progress(progress_val, desc=f"Processing record {idx + 1} of {total_records}...")
        
        try:
            record = manager.record(identifier=record_id)
            file_list = record.get_filelist().json()
            for file_info in file_list["items"]:
                file_names.append(file_info["name"])
                file_record_mapping[file_info["name"]] = record_id
        except kadi_apy.lib.exceptions.KadiAPYInputError as e:
            raise gr.Error(e)

        file_num = record.get_number_files()

        per_page = 100  # default in kadi
        not_divisible = file_num % per_page
        page_num = file_num // per_page + (1 if not_divisible else 0)
        
        for p in range(1, page_num + 1):
            sub_progress = progress_val + ((p/page_num) * (1/total_records))
            progress(sub_progress, desc=f"Loading files from record {idx + 1}, page {p} of {page_num}...")
            for file_info in record.get_filelist(page=p, per_page=per_page).json()["items"]:
                file_names.append(file_info["name"])
                file_record_mapping[file_info["name"]] = record_id

    progress(1.0, desc="Files loaded!")
    return file_names, f"Found {len(file_names)} files", file_record_mapping


def get_all_records(user_token, progress=gr.Progress()):
    """Get all record list in Kadi."""
    if not user_token:
        return [], "No user token found"

    progress(0, desc="Connecting to Kadi...")
    manager = KadiManager(instance=instance, host=host, pat=user_token)

    host_api = manager.host if manager.host.endswith("/") else manager.host + "/"
    searched_resource = "records"
    endpoint = urljoin(host_api, searched_resource)

    progress(0.2, desc="Searching records...")
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
        progress(0.2 + (0.8 * page/total_pages), desc=f"Loading page {page} of {total_pages}...")
        page_endpoint = endpoint + f"?page={page}&per_page=100"
        response = manager.make_request(page_endpoint)
        parsed = json.loads(response.content)
        all_records_identifiers.extend(get_page_records(parsed))

    progress(1.0, desc="Records loaded!")
    return all_records_identifiers, f"Found {len(all_records_identifiers)} records"


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
        # print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        if self.embeddings_model is None:
            self.embeddings_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2", trust_remote_code=True
            )

        print("Getting document embeddings...")
        
        # Process documents in batches
        batch_size = 32  # Adjust based on your memory constraints
        num_documents = len(self.documents)
        all_embeddings = []
        
        for i in range(0, num_documents, batch_size):
            batch_docs = [doc["content"] for doc in self.documents[i:i + batch_size]]
            print(f"Processing batch {i//batch_size + 1} of {(num_documents + batch_size - 1)//batch_size}")
            
            # Use local model instead of API for faster processing
            embeddings = self.embeddings_model.encode(
                batch_docs,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            all_embeddings.append(embeddings)
        
        # Combine all batches
        self.embeddings = np.vstack(all_embeddings)
        
        # Build the FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        print(f"Vector database built successfully! Index dimension: {self.index.d}")

    def search_documents(self, query: str, k: int = 4) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        
        print("Getting query embedding...")
        # Use local model for consistency with build_vector_db
        query_embedding = self.embeddings_model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Index dimension: {self.index.d}")
        
        # Ensure dimensions match
        if query_embedding.shape[1] != self.index.d:
            raise ValueError(f"Embedding dimension {query_embedding.shape[1]} does not match index dimension {self.index.d}")
        
        D, I = self.index.search(query_embedding, k)
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
    """Extracts text from a PDF file and stores it in the property documents by page."""

    doc = pymupdf.open(file_path)
    documents = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        documents.append({"page": page_num + 1, "content": text})
    print("PDF processed successfully!")
    return documents


def prepare_file_for_chat(all_records_identifiers, file_names, token, file_record_mapping, progress=gr.Progress()):
    """Parse file and prepare RAG."""
    if not all_records_identifiers:
        raise gr.Error("No records found")
    
    if not file_names:
        raise gr.Error("No files found in the selected records")

    documents = []
    total_files = len(file_names)
    
    for fidx, file_name in enumerate(file_names):
        # Skip non-PDF files
        if not file_name.lower().endswith('.pdf'):
            print(f"Skipping non-PDF file: {file_name}")
            continue
            
        current_progress = 0.1 + (0.4 * (fidx/total_files))
        progress(current_progress, desc=f"Processing {file_name}")
        
        # Get the record ID directly from the mapping
        record_id = file_record_mapping.get(file_name)
        if record_id:
            manager = KadiManager(instance=instance, host=host, pat=token)
            record = manager.record(identifier=record_id)
            
            try:
                file_id = record.get_file_id(file_name)
                with tempfile.TemporaryDirectory(prefix="tmp-kadichat-downloads-") as temp_dir:
                    temp_file_location = os.path.join(temp_dir, file_name)
                    record.download_file(file_id, temp_file_location)
                    docs = load_and_chunk_pdf(temp_file_location)
                    documents.extend(docs)
            except kadi_apy.lib.exceptions.KadiAPYInputError as e:
                print(f"Error processing file {file_name}: {e}")
        else:
            print(f"Warning: No record mapping found for file {file_name}")

    if not documents:
        raise gr.Error("No PDF documents could be processed")

    progress(0.5, desc="Initializing embeddings model...")
    user_rag = SimpleRAG()
    user_rag.documents = documents
    user_rag.embeddings_model = embeddings_model
    
    progress(0.7, desc="Building vector database...")
    user_rag.build_vector_db()
    
    progress(1.0, desc="Ready to chat!")
    return "Ready to chat! You can now ask questions about your documents.", user_rag


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
                get_files_btn = gr.Button("Get All Files")
                status_box = gr.Textbox(
                    label="Status", value="Click 'Get All Files' to start", interactive=False
                )
                message_box = gr.Textbox(
                    label="Chat Status", value="", interactive=False
                )

                # Change these from lists to Gradio components
                record_list = gr.State([])  # Using gr.State to store the records
                record_file_dropdown = gr.State([])  # Using gr.State to store the files

                # Initialize user token and get records list
                main_demo.load(_init_user_token, None, _state_user_token)

                # Create chain of events when Get All Files is clicked
                get_files_btn.click(
                    fn=get_all_records,
                    inputs=[_state_user_token],
                    outputs=[record_list, status_box]
                ).success(
                    fn=get_files_in_record,
                    inputs=[record_list, _state_user_token],
                    outputs=[record_file_dropdown, status_box, file_record_mapping]
                ).success(
                    fn=prepare_file_for_chat,
                    inputs=[record_list, record_file_dropdown, _state_user_token, file_record_mapping],
                    outputs=[message_box, user_session_rag]
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