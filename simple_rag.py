from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb import Chroma
from document_processor import DocumentProcessor

class SimpleRAG:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None
        self.documents = []
        
    def add_documents(self, documents: List[str]) -> int:
        """Process and add documents to the RAG system."""
        try:
            processed_chunks = []
            for doc in documents:
                clean_text = self.document_processor.preprocess_document(doc)
                chunks = self.document_processor.chunk_document(clean_text)
                processed_chunks.extend(chunks)
            
            self.documents = documents
            
            if not self.vector_store:
                self.vector_store = Chroma(
                    collection_name="documents",
                    embedding_function=self.embeddings
                )
            
            self.vector_store.add_texts(
                texts=processed_chunks,
                metadatas=[{
                    "chunk_id": i,
                    "doc_id": f"doc_{i//5}"
                } for i in range(len(processed_chunks))]
            )
            
            return len(processed_chunks)
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return 0
    
    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant document chunks."""
        try:
            if not self.vector_store:
                return []
            
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            threshold = 0.7
            filtered_results = []
            seen_doc_ids = set()
            
            for doc, score in results:
                if score > threshold:
                    doc_id = doc.metadata.get('doc_id')
                    if doc_id not in seen_doc_ids:
                        filtered_results.append(doc.page_content)
                        seen_doc_ids.add(doc_id)
            
            return filtered_results
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    def get_context_for_query(self, query: str) -> str:
        """Get formatted context for a query."""
        try:
            relevant_chunks = self.search_documents(query)
            if not relevant_chunks:
                return ""
            
            context = "\n---\n".join(relevant_chunks)
            if len(context) > 2000:
                context = context[:1997] + "..."
            
            return context
            
        except Exception as e:
            print(f"Error getting context: {str(e)}")
            return ""
    
    def is_initialized(self) -> bool:
        """Check if the RAG system is properly initialized."""
        return self.vector_store is not None and len(self.documents) > 0