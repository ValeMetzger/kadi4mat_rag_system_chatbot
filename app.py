import gradio as gr
from typing import List, Tuple
from huggingface_hub import InferenceClient
from simple_rag import SimpleRAG
from document_processor import DocumentProcessor
from utils import clean_response

# Initialize the LLM client
client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token="your_huggingface_token"  # Replace with your token
)

# Initialize state
loading_state = gr.State(True)

def respond(message: str, history: List[Tuple[str, str]], user_session_rag):
    """Handle chat interactions."""
    if not user_session_rag or not user_session_rag.is_initialized():
        return (
            history + [
                (
                    message,
                    "Still loading documents or no documents found. Please wait a moment and try again.",
                )
            ],
            ""
        )

    try:
        # Get context for query
        context = user_session_rag.get_context_for_query(message)
        if not context:
            return history + [(message, "No relevant information found in the documents.")], ""
        
        # Format prompt
        prompt = f"""<s>[INST] You are a helpful assistant. Please provide a clear and concise response.

Context (relevant sections from documents):
{context}

User Question: {message}

Important: Focus only on information from the provided context. If the context doesn't contain relevant information, say so. [/INST]"""
        
        # Generate response
        response = client.text_generation(
            prompt,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            stop_sequences=["</s>", "[INST]"]
        )
        
        # Clean and return response
        cleaned_response = clean_response(response)
        return history + [(message, cleaned_response)], ""
        
    except Exception as e:
        return history + [(message, f"Error: {str(e)}")], ""

async def process_all_records_and_files(user_token: str) -> List[str]:
    """Process and return all documents for the user."""
    # Implement your document loading logic here
    # This should return a list of document strings
    pass

def create_demo() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    with gr.Blocks() as demo:
        # Loading status
        loading_status = gr.Markdown("Initializing system and loading records...")
        
        # Chat interface
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            height=500
        )
        
        # Input components
        msg = gr.Textbox(
            label="Chat Message",
            placeholder="Type your message here...",
            lines=1
        )
        clear = gr.Button("Clear")
        
        # Set up event handlers
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        clear.click(lambda: None, None, chatbot, queue=False)
        
        # Initialize RAG system on load
        @demo.load
        async def initialize_rag(request: gr.Request):
            try:
                user_token = request.request.session["user_access_token"]
                
                # Create RAG instance
                rag_system = SimpleRAG()
                
                # Get and process documents
                documents = await process_all_records_and_files(user_token)
                num_chunks = rag_system.add_documents(documents)
                
                loading_state.value = False
                return gr.Markdown(value=f"Ready to chat! Processed {len(documents)} documents into {num_chunks} chunks.")
            except Exception as e:
                loading_state.value = False
                return gr.Markdown(value=f"Error loading records: {str(e)}")
        
        return demo

# Create and launch the demo
demo = create_demo()

if __name__ == "__main__":
    demo.launch()