import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import pickle
from pathlib import Path
import sys
import os
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration,      # Specific T5 model class
    T5Tokenizer                      # Specific T5 tokenizer
)


# Add the directory containing financial-preprocessor.py to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))    

class RAGSystem:
    def __init__(self, 
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 llm_model_name: str = 'google/flan-t5-base',
                 index_path: str = 'faiss_index',
                 chunks_path: str = 'chunks.pkl'):
        """
        Initialize the RAG system with embedding model and vector store
        
        Args:
            embedding_model_name: Name of the sentence-transformers model to use
            llm_model_name: Name of the language model to use
            index_path: Path to save/load FAISS index
            chunks_path: Path to save/load text chunks
        """
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize language model and tokenizer
        print("Loading language model...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(llm_model_name)
        
        # Initialize FAISS index
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.index = None
        self.chunks = None
        
        # Create or load index
        if self.index_path.exists() and self.chunks_path.exists():
            self.load_index()
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = self.embedding_model.encode(texts, 
                                               show_progress_bar=True,
                                               convert_to_numpy=True)
        return embeddings
    
    def build_index(self, chunks: List[str]):
        """Build FAISS index from text chunks"""
        print("Generating embeddings for chunks...")
        embeddings = self.embed_texts(chunks)
        
        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        self.chunks = chunks
        
        # Save index and chunks
        self.save_index()
        print("Index built and saved successfully!")
    
    def save_index(self):
        """Save FAISS index and chunks to disk"""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
    
    def load_index(self):
        """Load FAISS index and chunks from disk"""
        self.index = faiss.read_index(str(self.index_path))
        with open(self.chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query text
            k: Number of chunks to retrieve
            
        Returns:
            List of dictionaries containing chunks and their scores
        """
        # Generate query embedding
        query_embedding = self.embed_texts([query])
        
        # Search index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid index
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(1 / (1 + dist)),  # Convert distance to similarity score
                    'rank': i + 1
                })
        
        return results
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using the language model
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        # Prepare prompt
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=150,
            min_length=30,
            num_beams=4,
            no_repeat_ngram_size=3,
            #temperature=0.7
        )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a query through the full RAG pipeline
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            Dictionary containing response and retrieval details
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query, k)
        
        # Combine chunks for context
        context = "\n".join([chunk['chunk'] for chunk in retrieved_chunks])
        
        # Generate response
        response = self.generate_response(query, context)
        
        return {
            'response': response,
            'retrieved_chunks': retrieved_chunks,
            'confidence_score': np.mean([chunk['score'] for chunk in retrieved_chunks])
        }

def main():
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Load preprocessed chunks (from your previous preprocessing step)
    from financial_preprocessor import FinancialDataPreprocessor
    preprocessor = FinancialDataPreprocessor()
    file_paths = ['fin_stmts/goog-10-k-2024.pdf']
    chunks, _ = preprocessor.process_financial_statements(file_paths)
    
    # Build index
    rag.build_index(chunks)
    
    # Test query
    test_query = "What was the total revenue for 2024?"
    result = rag.query(test_query)
    
    print("\nTest Query:", test_query)
    print("\nResponse:", result['response'])
    print("\nConfidence Score:", result['confidence_score'])
    print("\nTop Retrieved Chunks:")
    for chunk in result['retrieved_chunks']:
        print(f"\nRank {chunk['rank']} (Score: {chunk['score']:.3f}):")
        print(chunk['chunk'])

if __name__ == "__main__":
    main()
