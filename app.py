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
        """Builds a vector database with dimension debugging."""
        if self.embeddings_model is None:
            self.embeddings_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2", 
                trust_remote_code=True
            )

        # Debug document count
        print(f"Number of documents to embed: {len(self.documents)}")
        
        # Get embeddings
        self.embeddings = self.embeddings_model.encode(
            [doc["content"] for doc in self.documents],
            show_progress_bar=True
        )
        
        # Debug embedding dimensions
        print(f"Embeddings shape: {self.embeddings.shape}")
        print(f"Expected dimension: 768")  # mpnet-base-v2 dimension
        
        # Initialize FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        print(f"FAISS index dimension: {self.index.d}")
        
        # Add vectors to index
        self.index.add(np.array(self.embeddings))
        print(f"Total vectors in index: {self.index.ntotal}")

    def search_documents(self, query: str, k: int = 4) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        
        # Get query embedding
        embedding_responses = embeddings_client.post(
            json={"inputs": [query]}, task="feature-extraction"
        )
        query_embedding = json.loads(embedding_responses.decode())
        
        # Debug print
        print(f"Query embedding shape: {np.array(query_embedding).shape}")
        
        # Handle the (1,1,8,768) shape by taking the first embedding
        query_embedding = np.array(query_embedding)
        if len(query_embedding.shape) > 2:
            # Take the first embedding (averaging might be another option)
            query_embedding = query_embedding[0, 0, 0].reshape(1, -1)
        
        print(f"Processed query shape: {query_embedding.shape}")
        print(f"Index dimension: {self.index.d}")
        
        # Verify dimensions
        assert query_embedding.shape[1] == self.index.d, \
            f"Query dimension {query_embedding.shape[1]} != Index dimension {self.index.d}"
        
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


def prepare_file_for_chat(all_records_identifiers, all_file_names, token, progress=gr.Progress()):
    """Parse file and prepare RAG."""

    if not all_file_names:
        raise gr.Error("No files found")
    progress(0, desc="Starting")
    
    # Create connection to kadi
    manager = KadiManager(instance=instance, host=host, pat=token)
    documents = []
    
    total_files = len(all_file_names)
    files_processed = 0
    
    # Iterate through all records
    for record_id in all_records_identifiers:
        try:
            record = manager.record(identifier=record_id)
            
            # Get all files for this record
            record_files = record.get_filelist().json()["items"]
            
            # Process each file in this record
            for file_info in record_files:
                file_name = file_info["name"]
                if file_name in all_file_names:  # Only process files we want
                    progress(0.2 + (0.6 * files_processed/total_files), 
                            desc=f"Processing {file_name}...")
                    
                    file_id = file_info["id"]
                    with tempfile.TemporaryDirectory(prefix="tmp-kadichat-downloads-") as temp_dir:
                        temp_file_location = os.path.join(temp_dir, file_name)
                        record.download_file(file_id, temp_file_location)
                        
                        # Parse document
                        try:
                            docs = load_and_chunk_pdf(temp_file_location)
                            # Add source information to each chunk
                            for doc in docs:
                                doc["metadata"].update({
                                    "file_name": file_name,
                                    "record_id": record_id
                                })
                            documents.extend(docs)
                        except Exception as e:
                            print(f"Error processing file {file_name}: {e}")
                            continue
                            
                    files_processed += 1
                    
        except kadi_apy.lib.exceptions.KadiAPYInputError as e:
            print(f"Error accessing record {record_id}: {e}")
            continue

    if not documents:
        raise gr.Error("No documents were successfully processed")

    progress(0.8, desc="Building vector database...")
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