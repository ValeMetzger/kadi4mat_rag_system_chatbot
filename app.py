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
        self.retrieval_metrics = []
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
        """Builds a vector database using the content of the documents."""
        if self.embeddings_model is None:
            self.embeddings_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2", trust_remote_code=True
            )

        print("Getting document embeddings...")
        
        # Check if there are documents to process
        if not self.documents:
            print("No documents to process.")
            return
        
        # Process documents in batches
        batch_size = 32  # Adjust based on your memory constraints
        num_documents = len(self.documents)
        all_embeddings = []
        
        try:
            for i in range(0, num_documents, batch_size):
                batch_docs = [doc["content"] for doc in self.documents[i:i + batch_size]]
                print(f"Processing batch {i//batch_size + 1} of {(num_documents + batch_size - 1)//batch_size}")
                
                # Filter out empty documents
                batch_docs = [doc for doc in batch_docs if doc.strip()]
                if not batch_docs:
                    print(f"Skipping batch {i//batch_size + 1} - all documents are empty")
                    continue
                
                # Use local model instead of API for faster processing
                embeddings = self.embeddings_model.encode(
                    batch_docs,
                    convert_to_numpy=True,
                    show_progress_bar=True
                )
                all_embeddings.append(embeddings)
            
            if not all_embeddings:
                print("No valid embeddings generated.")
                return
            
            # Combine all batches
            self.embeddings = np.vstack(all_embeddings)
            
            # Build the FAISS index
            print("Building FAISS index...")
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)
            print(f"Vector database built successfully! Index dimension: {self.index.d}, with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error building vector database: {e}")

    def search_documents(self, query: str, k: int = 4) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        
        start_time = time.time()
        
        if not self.documents or not self.index:
            return ["No documents have been processed yet."]
        
        try:
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
                print(f"WARNING: Embedding dimension mismatch: {query_embedding.shape[1]} vs {self.index.d}")
                return ["Error: Embedding dimension mismatch. Please rebuild the vector database."]
            
            D, I = self.index.search(query_embedding, k)
            
            # Check if any valid indices were returned
            valid_indices = [i for i in I[0] if 0 <= i < len(self.documents)]
            if not valid_indices:
                return ["No relevant documents found."]
            
            results = [self.documents[i]["content"] for i in valid_indices]
            
            # Track retrieval metrics
            retrieval_time = time.time() - start_time
            self.retrieval_metrics.append({
                "query": query,
                "num_results": len(results),
                "retrieval_time_ms": retrieval_time * 1000,
                "distances": D[0].tolist() if len(D) > 0 else [],
                "indices": I[0].tolist() if len(I) > 0 else []
            })
            
            return results
        except Exception as e:
            print(f"Error during document search: {e}")
            return [f"Error searching documents: {str(e)}"]

    def get_retrieval_metrics(self):
        """Returns the collected retrieval metrics."""
        return self.retrieval_metrics


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
    """Extracts text from a PDF file and chunks it."""
    try:
        with pymupdf.open(file_path) as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()

            if not text.strip():
                return [{"content": "The PDF appears to be empty or contains no extractable text.", 
                         "metadata": {"type": "pdf", "status": "empty"}}]

            chunks = chunk_text(text)
            documents = []
            for chunk in chunks:
                documents.append({"content": chunk, "metadata": {"type": "pdf"}})

            return documents
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return [{"content": f"Error processing PDF: {str(e)}", 
                 "metadata": {"type": "pdf", "status": "error"}}]


def load_text_file(file_path):
    """Extracts text from txt, md, or csv files and chunks it."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        chunks = chunk_text(text)
        documents = []
        for chunk in chunks:
            documents.append({"content": chunk, "metadata": {"type": "text"}})
        return documents


def load_docx(file_path):
    """Extracts text from docx files and chunks it."""
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        if not text.strip():
            return [{"content": "The DOCX file appears to be empty or contains no text.", 
                     "metadata": {"type": "docx", "status": "empty"}}]
        
        chunks = chunk_text(text)
        documents = []
        for chunk in chunks:
            documents.append({"content": chunk, "metadata": {"type": "docx"}})
        return documents
    except ImportError:
        print("python-docx package is not installed. Cannot process DOCX files.")
        return [{"content": "DOCX processing is not available. Please install python-docx package.", 
                 "metadata": {"type": "docx", "status": "error"}}]
    except Exception as e:
        print(f"Error processing DOCX {file_path}: {e}")
        return [{"content": f"Error processing DOCX: {str(e)}", 
                 "metadata": {"type": "docx", "status": "error"}}]


def load_csv_file(file_path):
    """Extracts text from CSV files with special handling."""
    try:
        import csv
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)
            
            if not rows:
                return [{"content": "The CSV file appears to be empty.", 
                         "metadata": {"type": "csv", "status": "empty"}}]
            
            # Convert CSV to a more readable format
            text = "\n".join([", ".join(row) for row in rows])
            chunks = chunk_text(text)
            documents = []
            for chunk in chunks:
                documents.append({"content": chunk, "metadata": {"type": "csv"}})
            return documents
    except Exception as e:
        print(f"Error processing CSV {file_path}: {e}")
        return [{"content": f"Error processing CSV: {str(e)}", 
                 "metadata": {"type": "csv", "status": "error"}}]


def prepare_file_for_chat(all_records_identifiers, file_names, token, file_record_mapping, progress=gr.Progress()):
    """Parse file and prepare RAG."""
    if not all_records_identifiers:
        raise gr.Error("No records found")
    
    if not file_names:
        raise gr.Error("No files found in the selected records")

    documents = []
    total_files = len(file_names)
    
    supported_extensions = {'.pdf', '.txt', '.md', '.csv', '.docx'}
    skipped_files = []
    processed_files = []
    
    for fidx, file_name in enumerate(file_names):
        file_ext = os.path.splitext(file_name.lower())[1]
        
        # Skip unsupported files
        if file_ext not in supported_extensions:
            skipped_files.append(file_name)
            print(f"Skipping unsupported file: {file_name}")
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
                    
                    # Process different file types
                    if file_ext == '.pdf':
                        docs = load_and_chunk_pdf(temp_file_location)
                    elif file_ext == '.docx':
                        docs = load_docx(temp_file_location)
                    elif file_ext == '.csv':
                        docs = load_csv_file(temp_file_location)
                    else:  # .txt, .md
                        docs = load_text_file(temp_file_location)
                        
                    documents.extend(docs)
            except kadi_apy.lib.exceptions.KadiAPYInputError as e:
                print(f"Error processing file {file_name}: {e}")
        else:
            print(f"Warning: No record mapping found for file {file_name}")

        processed_files.append(file_name)

    # Provide feedback about skipped files
    if skipped_files:
        message = f"Processed {len(processed_files)} files. Skipped {len(skipped_files)} unsupported files: {', '.join(skipped_files[:5])}"
        if len(skipped_files) > 5:
            message += f" and {len(skipped_files) - 5} more."
    else:
        message = f"Processed all {len(processed_files)} files successfully."

    if not documents:
        raise gr.Error("No documents could be processed")

    progress(0.5, desc="Initializing embeddings model...")
    user_rag = SimpleRAG()
    user_rag.documents = documents
    user_rag.embeddings_model = embeddings_model
    
    progress(0.7, desc="Building vector database...")
    user_rag.build_vector_db()
    
    progress(1.0, desc="Ready to chat!")
    return message, user_rag


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


def respond(message: str, history: List[Tuple[str, str]], user_session_rag, evaluate_mode=False, ground_truth=None):
    """Get respond from LLMs with evaluation metrics."""
    
    start_time = time.time()
    
    # RAG
    retrieved_docs = user_session_rag.search_documents(message)
    context = "\n".join(retrieved_docs)
    system_message = "You are an assistant to help user to answer question related to Kadi based on Relevant documents.\nRelevant documents: {}".format(
        context
    )
    messages = [{"role": "assistant", "content": system_message}]
    
    # Add history for conversational chat
    messages.append({"role": "user", "content": f"\nQuestion: {message}"})
    
    # Get answer from LLM
    response = client.chat_completion(
        messages, max_tokens=2048, temperature=0.0
    )
    response_content = "".join(
        [
            choice.message["content"]
            for choice in response.choices
            if "content" in choice.message
        ]
    )
    
    # Process response
    polished_response = preprocess_response(response_content)
    
    # Calculate metrics
    end_time = time.time()
    response_time = end_time - start_time
    
    # Estimate token count (rough approximation)
    input_tokens = len(message.split()) + len(context.split())
    output_tokens = len(polished_response.split())
    
    # Store metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "query": message,
        "response_time_seconds": response_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "num_retrieved_docs": len(retrieved_docs)
    }
    
    evaluation_metrics["response_times"].append(response_time)
    evaluation_metrics["token_counts"].append({"input": input_tokens, "output": output_tokens})
    
    # If in evaluation mode, check against ground truth
    if evaluate_mode and ground_truth:
        # Simple hallucination check - see if response contains information not in retrieved docs
        hallucination_score = check_hallucination(polished_response, retrieved_docs, ground_truth)
        metrics["hallucination_score"] = hallucination_score
        evaluation_metrics["hallucination_checks"].append(hallucination_score)
    
    # Save metrics to file for later analysis
    try:
        with open("evaluation_metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
    except Exception as e:
        print(f"Error saving metrics: {e}")
    
    history.append((message, polished_response))
    
    if evaluate_mode:
        return history, "", metrics
    else:
        return history, ""

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
                file_record_mapping = gr.State({})  # Add this line to create state for file mapping

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
            ["Summarize the document <Document_name>"],
            ["Can you tell me something about this record <Record_name>"],
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

    with gr.Tab("Evaluation"):
        with gr.Row():
            with gr.Column(scale=7):
                eval_chatbot = gr.Chatbot()
            with gr.Column(scale=3):
                gr.Markdown("### Evaluation Settings")
                eval_query = gr.Textbox(label="Test Query")
                ground_truth_docs = gr.Textbox(
                    label="Ground Truth Document IDs (comma-separated)",
                    placeholder="doc1,doc2,doc3"
                )
                key_facts = gr.Textbox(
                    label="Key Facts (one per line)",
                    placeholder="Fact 1\nFact 2\nFact 3",
                    lines=5
                )
                run_eval_btn = gr.Button("Run Evaluation")
                
                eval_results = gr.JSON(label="Evaluation Results")
                
                export_metrics_btn = gr.Button("Export All Metrics")
                metrics_output = gr.File(label="Exported Metrics")
        
        with gr.Row():
            retrieval_metrics_chart = gr.Plot(label="Retrieval Performance")
            response_time_chart = gr.Plot(label="Response Times")
        
        # Function to run evaluation
        def run_evaluation(query, ground_truth_doc_ids, key_facts_text, history, user_rag):
            # Parse ground truth inputs
            doc_ids = [doc_id.strip() for doc_id in ground_truth_doc_ids.split(",") if doc_id.strip()]
            facts = [fact.strip() for fact in key_facts_text.split("\n") if fact.strip()]
            
            ground_truth = {
                "document_ids": doc_ids,
                "key_facts": facts
            }
            
            # Run response with evaluation
            new_history, _, metrics = respond(
                query, 
                history if history else [], 
                user_rag,
                evaluate_mode=True,
                ground_truth=ground_truth
            )
            
            # Get retrieval metrics
            retrieved_docs = user_rag.search_documents(query)
            retrieval_quality = evaluate_retrieval_quality(query, retrieved_docs, doc_ids)
            
            # Combine metrics
            combined_metrics = {**metrics, **retrieval_quality}
            
            # Generate charts
            retrieval_chart = generate_retrieval_chart(user_rag.get_retrieval_metrics())
            response_chart = generate_response_time_chart(evaluation_metrics["response_times"])
            
            return new_history, combined_metrics, retrieval_chart, response_chart
        
        # Function to export all metrics
        def export_all_metrics():
            metrics_file = "all_evaluation_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(evaluation_metrics, f, indent=2)
            return metrics_file
        
        # Connect buttons to functions
        run_eval_btn.click(
            fn=run_evaluation,
            inputs=[eval_query, ground_truth_docs, key_facts, eval_chatbot, user_session_rag],
            outputs=[eval_chatbot, eval_results, retrieval_metrics_chart, response_time_chart]
        )
        
        export_metrics_btn.click(
            fn=export_all_metrics,
            inputs=[],
            outputs=[metrics_output]
        )

# Helper functions for visualization
def generate_retrieval_chart(retrieval_metrics):
    """Generate a chart showing retrieval performance metrics."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not retrieval_metrics or len(retrieval_metrics) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No retrieval data available", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Extract data
    queries = [m.get("query", "")[:20] + "..." for m in retrieval_metrics]
    times = [m.get("retrieval_time_ms", 0) for m in retrieval_metrics]
    num_results = [m.get("num_results", 0) for m in retrieval_metrics]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot retrieval times
    color = 'tab:blue'
    ax1.set_xlabel('Queries')
    ax1.set_ylabel('Retrieval Time (ms)', color=color)
    ax1.bar(np.arange(len(queries)), times, color=color, alpha=0.7, label='Retrieval Time')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis for number of results
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Number of Results', color=color)
    ax2.plot(np.arange(len(queries)), num_results, color=color, marker='o', label='Number of Results')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Set x-ticks
    plt.xticks(np.arange(len(queries)), queries, rotation=45, ha='right')
    
    # Add title and adjust layout
    plt.title('Retrieval Performance Metrics')
    fig.tight_layout()
    
    return fig

def generate_response_time_chart(response_times):
    """Generate a chart showing response time trends."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not response_times or len(response_times) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No response time data available", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot response times
    ax.plot(np.arange(len(response_times)), response_times, marker='o', linestyle='-')
    
    # Add moving average if we have enough data points
    if len(response_times) >= 3:
        window_size = min(3, len(response_times))
        moving_avg = np.convolve(response_times, np.ones(window_size)/window_size, mode='valid')
        ax.plot(np.arange(len(moving_avg)) + window_size-1, moving_avg, 'r--', label=f'{window_size}-point Moving Average')
    
    # Add labels and title
    ax.set_xlabel('Query Number')
    ax.set_ylabel('Response Time (seconds)')
    ax.set_title('Response Time Trend')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    if len(response_times) >= 3:
        ax.legend()
    
    return fig

app = gr.mount_gradio_app(app, main_demo, path="/gradio", auth_dependency=get_user)


def evaluate_retrieval_quality(query, retrieved_docs, ground_truth_docs, progress=gr.Progress()):
    """
    Evaluates the quality of retrieved documents against ground truth.
    
    Args:
        query: The user query
        retrieved_docs: List of retrieved document chunks
        ground_truth_docs: List of document IDs that should be retrieved
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    progress(0.1, desc="Evaluating retrieval quality...")
    
    # Extract document IDs from retrieved chunks (assuming metadata contains doc_id)
    retrieved_ids = set()
    for doc in retrieved_docs:
        if isinstance(doc, dict) and "metadata" in doc:
            if "doc_id" in doc["metadata"]:
                retrieved_ids.add(doc["metadata"]["doc_id"])
        elif isinstance(doc, str):
            # Try to extract document ID from content if available
            # This is a simplified approach - adjust based on your document structure
            for gt_id in ground_truth_docs:
                if gt_id in doc:
                    retrieved_ids.add(gt_id)
    
    progress(0.5, desc="Calculating metrics...")
    
    # Calculate precision, recall, F1
    ground_truth_set = set(ground_truth_docs)
    
    true_positives = len(retrieved_ids.intersection(ground_truth_set))
    
    precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
    recall = true_positives / len(ground_truth_set) if ground_truth_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    progress(1.0, desc="Evaluation complete!")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "retrieved_count": len(retrieved_ids),
        "ground_truth_count": len(ground_truth_set),
        "true_positives": true_positives
    }


if __name__ == "__main__":
    uvicorn.run(app, port=7860, host="0.0.0.0")