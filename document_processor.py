from typing import List
import re
import nltk
from transformers import AutoTokenizer

class DocumentProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        nltk.download('punkt', quiet=True)
    
    def preprocess_document(self, text: str) -> str:
        """Clean and preprocess document text."""
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def chunk_document(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split document into semantic chunks."""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer(sentence, return_length=True)
            if current_length + sentence_tokens['length'] > max_chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens['length']
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens['length']
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks