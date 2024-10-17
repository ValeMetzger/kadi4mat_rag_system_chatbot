"""
This is a demo to show how to use OAuth2 to connect an application to Kadi.

Read Section "OAuth2 Tokens" in Kadi documents.
Ref: https://kadi.readthedocs.io/en/stable/httpapi/intro.html#oauth2-tokens

Notes:
1. register an application in Kadi (Setting->Applications)
    - Name: KadiOAuthTest
    - Website URL: http://127.0.0.1:8000
    - Redirect URIs: http://localhost:8000/auth
    
And you will get Client ID and Client Secret, note them down and set in this file.

2. Start this app, and open browser with address "http://localhost:8000/"

"""

import json

import uvicorn
from fastapi import FastAPI, Depends
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import Request
import gradio as gr
import kadi_apy
from kadi_apy import KadiManager
from requests.compat import urljoin
from typing import List, Tuple
import pymupdf
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dotenv import load_dotenv
import os

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
instance = "my_kadi_demo_instance"  # "demo kit instance"
host = "https://demo-kadi4mat.iam.kit.edu"

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


embeddings_client = InferenceClient(model="sentence-transformers/all-mpnet-base-v2", token=huggingfacehub_api_token)
# embeddings_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True)  # unused
embeddings_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", trust_remote_code=True)

# Dependency to get the current user
def get_user(request: Request):
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
    root_url = gr.route_utils.get_root_url(request, "/", None)
    print("root url", root_url)
    if user:
        return RedirectResponse(url=f"{root_url}/gradio/")
    else:
        return RedirectResponse(url=f"{root_url}/main/")


@app.route("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    request.session.pop("user_id", None)
    request.session.pop("user_access_token", None)

    return RedirectResponse(url="/")


@app.route("/login")
async def login(request: Request):
    root_url = gr.route_utils.get_root_url(request, "/login", None)
    redirect_uri = request.url_for("auth")  # f"{root_url}/auth"
    redirect_uri = redirect_uri.replace(scheme='https')
    print("-----------in login")
    print("root_urlt", root_url)
    print("redirect_uri", redirect_uri)
    print("request", request)
    return await oauth.kadi4mat.authorize_redirect(request, redirect_uri)


@app.route("/auth")
async def auth(request: Request):
    root_url = gr.route_utils.get_root_url(request, "/auth", None)
    print("*****+ in auth")
    print("root_urlt", root_url)
    print("request", request)
    try:
        access_token = await oauth.kadi4mat.authorize_access_token(request)
        request.session["user_access_token"] = access_token["access_token"]

    except OAuthError as e:
        print("Error getting access token", e)
        return RedirectResponse(url="/")

    return RedirectResponse(url="/gradio")


def greet(request: gr.Request):
    return f"Welcome to Kadichat, you're logged in as: {request.username}"


def get_files_in_record(record_id, user_token, top_k=10):

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
    user_token = request.request.session["user_access_token"]
    return user_token


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

import tempfile
import os
import pymupdf

class SimpleRAG:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings_model = None
        self.embeddings = None
        self.index = None
        #self.load_pdf("Brandt et al_2024_Kadi_info_page.pdf")
        #self.build_vector_db()

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
        """Builds a vector database using the content of the PDF."""
        if self.embeddings_model is None:
            self.embeddings_model = SentenceTransformer("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)  # jinaai/jina-embeddings-v2-base-de?
        # Use embeddings_client
        print("now doing embedding")
        print("len of documents", len(self.documents))
        import time
        start =time.time()
        #embedding_responses = embeddings_client.post(json={"inputs":[doc["content"] for doc in self.documents]}, task="feature-extraction")
        #self.embeddings = np.array(json.loads(embedding_responses.decode()))
        self.embeddings = self.embeddings_model.encode([doc["content"] for doc in self.documents], show_progress_bar=True)
        end = time.time()
        print("cost time", end-start)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 4) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        # query_embedding = self.embeddings_model.encode([query], show_progress_bar=False)
        embedding_responses = embeddings_client.post(json={"inputs": [query]}, task="feature-extraction")
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
            while end > start and text[end] != '\n':
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

def load_pdf(file_path: str) -> None:
    """Extracts text from a PDF file and stores it in the property documents by page."""
    doc = pymupdf.open(file_path)
    documents = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        documents.append({"page": page_num + 1, "content": text})
    print("PDF processed successfully!")
    return documents
        
def prepare_file_for_chat(record_id, file_names, token, progress=gr.Progress()):
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
    # response = response.strip()
    # response = response.replace("\n\n", "\n")
    # response = response.replace(" ,", ",")
    # response = response.replace(" .", ".")
    # response = " ".join(response.split())
    # if not any(word in response.lower() for word in ["sorry", "apologize", "empathy"]):
    #     response = "I'm here to help. " + response
    return response


def respond(message: str, history: List[Tuple[str, str]], user_session_rag):
    
    # message is the current input query from user
    # RAG
    retrieved_docs = user_session_rag.search_documents(message)
    context = "\n".join(retrieved_docs)
    system_message = "You are an assistant to help user to answer question related to Kadi based on Relevant documents.\nRelevant documents: {}".format(context)
    messages = [{"role": "assistant", "content": system_message}]

    # Add history for conversational chat, TODO
    # for val in history:
    #     #if val[0]:
    #     messages.append({"role": "user", "content": val[0]})
    #     #if val[1]:
    #     messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": f"\nQuestion: {message}"})

    print("-----------------")
    print(messages)
    print("-----------------")
    # Get anwser from LLM
    response = client.chat_completion(messages, max_tokens=2048, temperature=0.0)  #, top_p=0.9)
    response_content = "".join([choice.message['content'] for choice in response.choices if 'content' in choice.message])
    
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

    user_session_rag = gr.State(
        "placeholder"#, time_to_live=3600
    )  # clean state after 1h
    
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
                record_file_dropdown = gr.Dropdown(
                    choices=[""],
                    label="Select file",
                    info="Select (max. 3) files to chat with.",
                    multiselect=True,
                    max_choices=3,
                )

                gr.Markdown("  " * 200)
                # Use .then to ensure get token first
                main_demo.load(_init_user_token, None, _state_user_token).then(
                    get_all_records, _state_user_token, record_list
                )

                parse_files = gr.Button("Parse files")
                # message_box = gr.Markdown("")
                message_box =  gr.Textbox(label="", value="progress bar", interactive=False)
                # Interactions
                # Update file list after selecting record
                record_list.select(
                    fn=get_files_in_record,
                    inputs=[record_list, _state_user_token],
                    outputs=record_file_dropdown,
                )
                # Prepare files for chatbot
                parse_files.click(fn=prepare_file_for_chat, inputs=[record_list, record_file_dropdown, _state_user_token], outputs=[message_box, user_session_rag])

        with gr.Row():
            txt_input = gr.Textbox(
                show_label=False,
                placeholder="Type your question here...",
                lines=1
            )
            submit_btn = gr.Button("Submit", scale=1)
            refresh_btn = gr.Button("Refresh Chat", scale=1, variant="secondary")

        example_questions = [
            ["Summarize the paper."],
            ["how to create record in kadi4mat?"],
        ]

        gr.Examples(examples=example_questions, inputs=[txt_input])

        txt_input.submit(fn=respond, inputs=[txt_input, chatbot, user_session_rag], outputs=[chatbot, txt_input])
        submit_btn.click(fn=respond, inputs=[txt_input, chatbot, user_session_rag], outputs=[chatbot, txt_input])
        refresh_btn.click(lambda: [], None, chatbot)

app = gr.mount_gradio_app(app, main_demo, path="/gradio", auth_dependency=get_user)


# def launch_gradio():
#    login_demo.launch(server_port=7860, host="0.0.0.0", share=True)


import threading

if __name__ == "__main__":
    # Launch Gradio with share=True in a separate thread
    # threading.Thread(target=launch_gradio).start()
    uvicorn.run(app, port=7860, host="0.0.0.0")
