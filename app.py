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
import time

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
        
    def add_documents(self, new_documents: List[dict]) -> None:
        """Add new documents to the existing collection"""
        self.documents.extend(new_documents)
        
    def build_vector_db(self) -> None:
        """Builds a vector database using all documents"""
        if not self.documents:
            print("No documents to build vector database")
            return
            
        # Use embeddings_client for consistency
        contents = [doc["content"] for doc in self.documents]
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            try:
                # Use feature-extraction endpoint correctly
                embedding_responses = embeddings_client.post(
                    json={"inputs": batch},
                    task="feature-extraction"
                )
                # The response is already a numpy array, no need for json.loads and decode
                batch_embeddings = np.array(embedding_responses)
                all_embeddings.append(batch_embeddings)
                print(f"Processed batch {i//batch_size + 1}/{(len(contents) + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
        
        if not all_embeddings:
            print("No embeddings were generated successfully")
            return
        
        try:
            self.embeddings = np.vstack(all_embeddings)
            
            # Initialize FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings)
            print(f"Vector database built successfully with {len(self.documents)} documents!")
        except Exception as e:
            print(f"Error building vector database: {str(e)}")
            return

    def search_documents(self, query: str, k: int = 4) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        if not self.index:
            return ["Vector database not initialized."]
            
        try:
            # Get query embedding using the same client
            embedding_response = embeddings_client.post(
                json={"inputs": query},
                task="feature-extraction"
            )
            # The response is already a numpy array
            query_embedding = np.array(embedding_response).reshape(1, -1)
            
            # Search similar documents
            D, I = self.index.search(query_embedding, k)
            
            # Add metadata to results
            results_with_metadata = []
            for idx in I[0]:
                doc = self.documents[idx]
                metadata_str = f"\nSource: Record {doc['metadata'].get('record_id', 'unknown')}, File: {doc['metadata'].get('file_name', 'unknown')}"
                results_with_metadata.append(doc["content"] + metadata_str)
            
            return results_with_metadata if results_with_metadata else ["No relevant documents found."]
        except Exception as e:
            print(f"Error during document search: {str(e)}")
            return ["Error occurred while searching documents."]


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
    """Extracts text from a PDF file and chunks it into documents."""
    try:
        with pymupdf.open(file_path) as pdf:
            text = ""
            for page in pdf:
                page_text = page.get_text()
                if page_text.strip():  # Only add non-empty pages
                    text += page_text + "\n\n"  # Add page breaks

            if not text.strip():
                print(f"Warning: No text content found in {file_path}")
                return []

            chunks = chunk_text(text)
            if not chunks:
                print(f"Warning: No chunks generated for {file_path}")
                return []

            documents = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    documents.append({
                        "content": chunk,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    })

            return documents

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
    """Initialize RAG system with all records from Kadi"""
    progress(0, desc="Starting RAG initialization")
    
    # Create connection to kadi
    manager = KadiManager(instance=instance, host=host, pat=token)
    
    # Initialize RAG
    rag_system = SimpleRAG()
    
    try:
        # Get all records
        print("Fetching records from Kadi...")
        response = manager.search.search_resources("record", per_page=100)
        parsed = json.loads(response.content)
        
        # Debug print
        print(f"Response status: {response.status_code}")
        print(f"Response content keys: {list(parsed.keys())}")
        
        items = parsed.get("items", [])
        print(f"Found {len(items)} records")
        
        if not items:
            print("No records found in response")
            return "No records found in Kadi", None
        
        progress(0.1, desc=f"Found {len(items)} records to process")
        processed_records = 0
        processed_files = 0
        skipped_files = 0
        
        # Process each record
        for item in items:
            record_id = item.get("identifier")
            if not record_id:
                print("Skipping record with no identifier")
                continue
                
            print(f"\nProcessing record: {record_id}")
            
            try:
                record = manager.record(identifier=record_id)
                
                # Get files for this record
                file_list_response = record.get_filelist()
                print(f"File list response status: {file_list_response.status_code}")
                
                file_list = file_list_response.json()
                files = file_list.get("items", [])
                print(f"Found {len(files)} files in record {record_id}")
                
                # Process each file
                for file_info in files:
                    file_name = file_info.get("name", "").lower()
                    file_id = file_info.get("id")
                    
                    if not all([file_name, file_id]):
                        print(f"Skipping file with incomplete info: {file_name}")
                        continue
                    
                    # Skip non-PDF files
                    if not file_name.endswith('.pdf'):
                        skipped_files += 1
                        print(f"Skipping non-PDF file: {file_name}")
                        continue
                        
                    print(f"Processing PDF file: {file_name}")
                    
                    try:
                        with tempfile.TemporaryDirectory(prefix="tmp-kadichat-") as temp_dir:
                            temp_file_location = os.path.join(temp_dir, file_name)
                            
                            # Download file
                            print(f"Downloading file to {temp_file_location}")
                            download_response = record.download_file(file_id, temp_file_location)
                            
                            if not os.path.exists(temp_file_location):
                                print(f"Error: File not downloaded to {temp_file_location}")
                                continue
                                
                            # Parse document
                            docs = load_and_chunk_pdf(temp_file_location)
                            if docs:
                                for doc in docs:
                                    doc["metadata"] = {
                                        "record_id": record_id,
                                        "file_name": file_name,
                                        "record_title": item.get("title", "Unknown")
                                    }
                                rag_system.add_documents(docs)
                                processed_files += 1
                                print(f"Successfully processed {file_name} - got {len(docs)} chunks")
                            else:
                                print(f"No valid content extracted from {file_name}")
                            
                    except Exception as e:
                        print(f"Error processing file {file_name}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error processing record {record_id}: {str(e)}")
                continue
            
            processed_records += 1
            progress(0.1 + (0.7 * processed_records/len(items)), 
                    desc=f"Processed {processed_records}/{len(items)} records. Files: {processed_files} processed, {skipped_files} skipped")
        
        # Build vector database if we have documents
        if rag_system.documents:
            print(f"\nBuilding vector database with {len(rag_system.documents)} documents...")
            progress(0.8, desc="Building vector database...")
            rag_system.build_vector_db()
            progress(1.0, desc="RAG system ready")
            return f"RAG system initialized with {len(rag_system.documents)} documents from {processed_records} records ({processed_files} files processed, {skipped_files} skipped)", rag_system
        else:
            print("No documents were processed successfully")
            return "No documents were processed successfully", None
            
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        return f"Failed to initialize RAG system: {str(e)}", None


def preprocess_response(response: str) -> str:
    """Preprocesses the response to make it more polished."""
    if not response or not response.strip():
        return "I apologize, but I couldn't generate a proper response."
        
    # Basic cleaning
    response = response.strip()
    
    # Remove multiple consecutive newlines
    while "\n\n\n" in response:
        response = response.replace("\n\n\n", "\n\n")
        
    # Ensure the response starts with a proper introduction if it doesn't have one
    if not any(word in response.lower() for word in ["sorry", "apologize", "based on", "according to", "i found", "i can"]):
        response = "Based on the available information, " + response
        
    return response


def respond(message: str, history: List[Tuple[str, str]], user_session_rag):
    """Get respond from LLMs."""
    
    try:
        # Check if RAG system is properly initialized
        if not user_session_rag or user_session_rag == "placeholder":
            return history + [("Error: RAG system not initialized properly. Please refresh the page.", "")], ""
            
        # Debug print for RAG system state
        print(f"RAG system documents count: {len(user_session_rag.documents)}")
        
        # Get relevant documents
        print(f"Searching for documents relevant to: {message}")
        retrieved_docs = user_session_rag.search_documents(message)
        print(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        if not retrieved_docs or retrieved_docs == ["No relevant documents found."]:
            return history + [(message, "I don't have any relevant information in my knowledge base to answer your question. Could you please try a different question?")], ""
        
        # Build context from retrieved documents
        context = "\n".join(retrieved_docs)
        print(f"Context length: {len(context)} characters")
        
        # Construct prompt
        system_message = f"""You are a helpful assistant with access to a knowledge base about Kadi records and documents.
Please answer the user's question based on the following relevant documents. If you can't find relevant information,
please say so honestly.

Relevant documents:
{context}"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ]
        
        # Get answer from LLM
        print("Requesting response from LLM...")
        response = client.chat_completion(
            messages,
            max_tokens=2048,
            temperature=0.0,  # Slightly increased for more natural responses
            top_p=0.9
        )
        
        # Extract response content
        response_content = ""
        for choice in response.choices:
            if "content" in choice.message:
                response_content += choice.message["content"]
                
        if not response_content.strip():
            print("Warning: Empty response from LLM")
            return history + [(message, "I apologize, but I wasn't able to generate a proper response. Please try rephrasing your question.")], ""
            
        print(f"Generated response length: {len(response_content)} characters")
        
        # Process response
        polished_response = preprocess_response(response_content)
        
        # Update history and return
        history.append((message, polished_response))
        return history, ""
        
    except Exception as e:
        print(f"Error in respond function: {str(e)}")
        return history + [(message, f"I apologize, but an error occurred while processing your request: {str(e)}")], ""


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

        # Add this check after RAG initialization
        main_demo.load(
            _init_user_token, 
            None, 
            _state_user_token
        ).then(
            initialize_rag_system,
            inputs=[_state_user_token],
            outputs=[status_box, user_session_rag]
        ).then(
            check_rag_system,
            inputs=[user_session_rag],
            outputs=None
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