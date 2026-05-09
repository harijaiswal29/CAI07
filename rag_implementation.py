import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer
import faiss
import pickle
from pathlib import Path
import sys
import os
import re
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration
)

# Add the directory containing financial-preprocessor.py to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))    

# Simple BM25-like implementation to avoid external dependencies
class SimpleBM25:
    def __init__(self, tokenized_chunks):
        self.tokenized_chunks = tokenized_chunks
        self.doc_freqs = {}
        self.doc_lens = [len(chunk) for chunk in tokenized_chunks]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0
        self.N = len(tokenized_chunks)
        
        # Calculate document frequencies
        for doc in tokenized_chunks:
            for token in set(doc):
                if token not in self.doc_freqs:
                    self.doc_freqs[token] = 0
                self.doc_freqs[token] += 1
    
    def get_scores(self, query_tokens):
        scores = [0] * self.N
        k1 = 1.5
        b = 0.75
        
        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
                
            df = self.doc_freqs[token]
            idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
            
            for i, doc in enumerate(self.tokenized_chunks):
                if token not in doc:
                    continue
                    
                tf = doc.count(token)
                doc_len = self.doc_lens[i]
                
                # BM25 scoring formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / self.avg_doc_len)
                scores[i] += idf * numerator / denominator
                
        return scores

# Embedded Advanced Retrieval for better integration
class AdvancedRetrieval:
    """
    Advanced retrieval techniques for RAG systems
    """
    def __init__(self):
        """Initialize the advanced retrieval system"""
        self.bm25_index = None
        self.tokenized_chunks = None
        print("Initialized AdvancedRetrieval successfully!")
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        return re.findall(r'\b\w+\b', text.lower())
        
    def initialize_bm25(self, chunks: List[str]):
        """Initialize BM25 index from text chunks"""
        print(f"Initializing BM25 with {len(chunks)} chunks")
        self.tokenized_chunks = [self._tokenize(chunk) for chunk in chunks]
        self.bm25_index = SimpleBM25(self.tokenized_chunks)
        self.chunks = chunks
        print("BM25 index initialized successfully")
        
    def retrieve_with_bm25(self, query: str, chunks: List[str], top_k: int = 5) -> List[Dict]:
        """Retrieve chunks using BM25 scoring"""
        if self.bm25_index is None:
            self.initialize_bm25(chunks)
            
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k chunks
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Format results
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'chunk': chunks[idx],
                'score': float(bm25_scores[idx]),
                'rank': i + 1,
                'method': 'bm25'
            })
            
        return results
    
    def hybrid_retrieval(self, query: str, chunks: List[str], embedding_results: List[Dict], 
                         alpha: float = 0.5, top_k: int = 5) -> List[Dict]:
        """Combine embedding-based and BM25 retrieval results"""
        print(f"Running hybrid retrieval with alpha={alpha}")
        # Get BM25 results
        bm25_results = self.retrieve_with_bm25(query, chunks, top_k=top_k)
        
        # Create chunk-to-score mappings
        embedding_scores = {r['chunk']: r['score'] for r in embedding_results}
        bm25_scores = {r['chunk']: r['score'] for r in bm25_results}
        
        # Normalize scores
        max_embedding = max(embedding_scores.values()) if embedding_scores else 1.0
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        
        # Combine results
        combined_scores = {}
        
        # Add all chunks from both result sets
        for chunk in set(list(embedding_scores.keys()) + list(bm25_scores.keys())):
            emb_score = embedding_scores.get(chunk, 0) / max_embedding
            bm25_score = bm25_scores.get(chunk, 0) / max_bm25
            combined_scores[chunk] = alpha * emb_score + (1 - alpha) * bm25_score
        
        # Sort by combined score
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        for i, (chunk, score) in enumerate(sorted_chunks):
            results.append({
                'chunk': chunk,
                'score': float(score),
                'rank': i + 1,
                'method': 'hybrid'
            })
            
        return results
    
    def merge_chunks(self, chunks: List[Dict], similarity_threshold: float = 0.7) -> List[Dict]:
        """Merge similar or adjacent chunks"""
        if not chunks:
            return []
            
        # Extract chunk texts and scores
        texts = [chunk['chunk'] for chunk in chunks]
        scores = [chunk['score'] for chunk in chunks]
        
        # Merge similar chunks using word overlap
        merged_chunks = []
        used_indices = set()
        
        for i in range(len(texts)):
            if i in used_indices:
                continue
                
            current_text = texts[i]
            current_score = scores[i]
            current_words = set(self._tokenize(current_text))
            
            merged_text = current_text
            merged_indices = [i]
            
            # Check for similar chunks to merge
            for j in range(i + 1, len(texts)):
                if j in used_indices:
                    continue
                    
                compare_text = texts[j]
                compare_words = set(self._tokenize(compare_text))
                
                # Calculate Jaccard similarity
                overlap = len(current_words.intersection(compare_words))
                union = len(current_words.union(compare_words))
                similarity = overlap / union if union > 0 else 0
                
                if similarity > similarity_threshold:
                    merged_text = f"{merged_text}\n\n{compare_text}"
                    merged_indices.append(j)
                    used_indices.add(j)
            
            # Add merged chunk to results
            used_indices.add(i)
            merged_chunks.append({
                'chunk': merged_text,
                'score': current_score,  # Keep original score of primary chunk
                'rank': chunks[i]['rank'],
                'method': chunks[i].get('method', 'merged') + '_merged',
                'merged_count': len(merged_indices)
            })
            
        print(f"Merged {len(chunks)} chunks into {len(merged_chunks)} chunks")
        return merged_chunks
    


   # Updated adaptive_retrieval method in the AdvancedRetrieval class

    def adaptive_retrieval(self, query: str, chunks: List[str], embeddings_fn, 
                        k_values: List[int] = [3, 5, 7],
                        alpha_values: List[float] = [0.3, 0.5, 0.7]) -> Tuple[List[Dict], Dict]:
        """Adaptive retrieval that determines the best parameters based on query"""
        print(f"Running adaptive retrieval for query: '{query}'")
        
        # Extract query features
        query_features = self._extract_query_features(query)
        
        # Select parameters based on query characteristics
        k = self._select_k(query_features, k_values)
        alpha = self._select_alpha(query_features, alpha_values)
        should_merge = self._should_merge_chunks(query_features)
        should_rerank = self._should_rerank(query_features)
        
        print(f"Selected parameters: k={k}, alpha={alpha}, should_merge={should_merge}, should_rerank={should_rerank}")
        
        # Set retrieval size - get more results if we'll rerank
        retrieval_k = max(10, k*2) if should_rerank else k
        
        # Get embedding results
        embedding_results = embeddings_fn(query, k=retrieval_k)
        
        # Get hybrid results
        results = self.hybrid_retrieval(
            query=query,
            chunks=chunks,
            embedding_results=embedding_results,
            alpha=alpha,
            top_k=retrieval_k
        )
        
        # Merge chunks if needed
        if should_merge:
            results = self.merge_chunks(results)
        
        # Apply re-ranking if needed
        if should_rerank:
            results = self.rerank_with_cross_encoder(query, results, k)
        
        # Return results and parameters
        params = {
            'k': k,
            'alpha': alpha,
            'merged': should_merge,
            'reranked': should_rerank
        }
        
        return results, params 
    


    
    # Updated _extract_query_features method in the AdvancedRetrieval class

    def _extract_query_features(self, query: str) -> Dict:
        """Extract features from query for adaptive retrieval"""
        features = {}
        
        # Store lowercased query for keyword matching
        features['query_lower'] = query.lower()
        
        # Query length
        features['length'] = len(query)
        features['word_count'] = len(self._tokenize(query))
        
        # Question words
        question_words = ['what', 'when', 'where', 'why', 'who', 'how', 'which']
        features['is_question'] = any(q in features['query_lower'] for q in question_words)
        
        # Financial keywords
        financial_keywords = [
            'revenue', 'profit', 'earnings', 'growth', 'margin', 'debt', 
            'assets', 'liabilities', 'cash', 'equity', 'income', 'expense',
            'quarter', 'annual', 'fiscal', 'year', 'ratio', 'balance'
        ]
        features['financial_keywords'] = sum(1 for kw in financial_keywords if kw in features['query_lower'])
        
        # Extract years mentioned
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        features['years_mentioned'] = len(years)
        
        # Specificity score
        features['specificity'] = (features['financial_keywords'] + features['years_mentioned']) / max(1, features['word_count'])
        
        return features

    
    def _select_k(self, query_features: Dict, k_values: List[int]) -> int:
        """Select the best k value based on query features"""
        if query_features['specificity'] > 0.3:
            return min(k_values)
        elif query_features['word_count'] < 5:
            return max(k_values)
        else:
            return k_values[len(k_values) // 2]
    
    def _select_alpha(self, query_features: Dict, alpha_values: List[float]) -> float:
        """Select the best alpha value based on query features"""
        if query_features['years_mentioned'] > 0 and query_features['financial_keywords'] > 1:
            return max(alpha_values)
        elif query_features['is_question'] and query_features['word_count'] > 7:
            return alpha_values[len(alpha_values) // 2]
        else:
            return min(alpha_values)
    
    def _should_merge_chunks(self, query_features: Dict) -> bool:
        """Determine if chunks should be merged based on query features"""
        return query_features['word_count'] > 7 and query_features['financial_keywords'] > 1
    

    # Add this function to the AdvancedRetrieval class in rag_implementation.py

    def rerank_with_cross_encoder(self, query: str, retrieved_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-rank retrieved chunks using a cross-encoder model for more accurate relevance scoring
        
        Args:
            query: The query string
            retrieved_chunks: List of retrieved chunks with scores
            top_k: Number of chunks to return after re-ranking
            
        Returns:
            List of re-ranked chunks with updated scores
        """
        try:
            from sentence_transformers import CrossEncoder
            
            # Initialize cross-encoder model if not already loaded.
            # device='cpu' forces CPU even when WSL2 exposes a small integrated GPU —
            # BGE reranker (~280 MB + activations) doesn't fit in 2 GiB GPU memory and
            # would silently fall back to embedding ranking on every query.
            if not hasattr(self, 'cross_encoder'):
                print("Loading cross-encoder model (BAAI/bge-reranker-base)...")
                self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base', device='cpu')
                print("Cross-encoder loaded successfully")

            # Prepare pairs for scoring (query + document pairs)
            pairs = [(query, chunk['chunk']) for chunk in retrieved_chunks]

            # Score pairs with cross-encoder
            raw_scores = self.cross_encoder.predict(pairs)

            # Plain sigmoid — no logit sharpening (the *2.0 multiplier pushed weak matches
            # close to 1.0 and made low-confidence answers look high-confidence).
            normalized_scores = [1 / (1 + np.exp(-score)) for score in raw_scores]
            
            # Update scores and add original scores for reference
            for i, chunk in enumerate(retrieved_chunks):
                chunk['original_score'] = chunk['score']  # Keep original score for reference
                chunk['raw_ce_score'] = float(raw_scores[i])  # Store raw cross-encoder score
                chunk['score'] = float(normalized_scores[i])  # Update with normalized score
                chunk['method'] = chunk.get('method', 'unknown') + '_reranked'  # Update method
            
            # Sort by new scores
            reranked_chunks = sorted(retrieved_chunks, key=lambda x: x['score'], reverse=True)[:top_k]
            
            # Update ranks
            for i, chunk in enumerate(reranked_chunks):
                chunk['rank'] = i + 1
                
            print(f"Re-ranked {len(retrieved_chunks)} chunks to {len(reranked_chunks)} chunks")
            print(f"Score range after normalization: {min([c['score'] for c in reranked_chunks]):.4f} to {max([c['score'] for c in reranked_chunks]):.4f}")
            return reranked_chunks
            
        except Exception as e:
            print(f"Warning: Re-ranking failed - {str(e)}. Using original ranking.")
            return retrieved_chunks[:top_k]

    # Add this function to determine when to apply re-ranking
    def _should_rerank(self, query_features: Dict) -> bool:
        """
        Determine if re-ranking should be applied based on query features
        """
        # Apply re-ranking for more complex financial queries
        if query_features['word_count'] > 8:
            return True
        
        # Apply for specific financial analysis questions
        financial_analysis_words = ['compare', 'trend', 'analysis', 'performance', 'ratio', 
                                'growth', 'decline', 'impact', 'effect', 'cause']
        if any(word in query_features.get('query_lower', '') for word in financial_analysis_words):
            return True
        
        # Apply for questions with multiple financial keywords
        if query_features['financial_keywords'] >= 2:
            return True
            
        return False







class RAGSystem:
    def __init__(self,
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 llm_model_name: str = 'google/flan-t5-large',
                 index_path: str = 'faiss_index',
                 chunks_path: str = 'chunks.pkl',
                 enable_advanced_retrieval: bool = False):
        """
        Initialize the RAG system with embedding model and vector store
        
        Args:
            embedding_model_name: Name of the sentence-transformers model to use
            llm_model_name: Name of the language model to use
            index_path: Path to save/load FAISS index
            chunks_path: Path to save/load text chunks
            enable_advanced_retrieval: Whether to use advanced retrieval techniques
        """
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize language model and tokenizer.
        # flan-t5-large is ~3 GB on first download; HuggingFace caches it after.
        print(f"Loading language model ({llm_model_name})... first run downloads ~3 GB.")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(llm_model_name)
        
        # Initialize FAISS index
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.index = None
        self.chunks = None
        
        # Create or load index (IndexFlatIP + L2-normalized embeddings = cosine similarity)
        if self.index_path.exists() and self.chunks_path.exists():
            self.load_index()
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Cross-encoder is lazy-loaded on first re-rank call
        self.cross_encoder = None
            
        # Initialize advanced retrieval if enabled
        self.enable_advanced_retrieval = enable_advanced_retrieval
        if enable_advanced_retrieval:
            try:
                # Use the embedded AdvancedRetrieval class
                self.advanced_retrieval = AdvancedRetrieval()
                print("Advanced retrieval features enabled!")
            except Exception as e:
                print(f"WARNING: Failed to initialize advanced retrieval: {str(e)}")
                self.enable_advanced_retrieval = False
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate L2-normalized embeddings (so inner product = cosine similarity)."""
        embeddings = self.embedding_model.encode(texts,
                                               show_progress_bar=True,
                                               convert_to_numpy=True,
                                               normalize_embeddings=True)
        return embeddings
    
    def build_index(self, chunks: List[str]):
        """Build FAISS index from text chunks"""
        print("Generating embeddings for chunks...")
        embeddings = self.embed_texts(chunks)
        
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        self.chunks = chunks
        
        # Save index and chunks
        self.save_index()
        
        # Initialize BM25 for advanced retrieval if enabled
        if self.enable_advanced_retrieval and hasattr(self, 'advanced_retrieval'):
            print("Initializing BM25 index...")
            self.advanced_retrieval.initialize_bm25(chunks)
            
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
        # Generate query embedding (already L2-normalized)
        query_embedding = self.embed_texts([query])

        # Search index — IndexFlatIP returns cosine similarity (higher = better)
        similarities, indices = self.index.search(query_embedding.astype('float32'), k)

        # Format results
        results = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx != -1:  # Valid index
                results.append({
                    'chunk': self.chunks[idx],
                    'score': max(0.0, float(sim)),  # cosine similarity, clamped to [0, 1]
                    'rank': i + 1,
                    'method': 'embedding'
                })

        return results
    
    def _rerank_with_cross_encoder(self, query: str, candidates: List[Dict[str, Any]],
                                   top_k: int) -> List[Dict[str, Any]]:
        """Re-rank candidates with a cross-encoder; score = sigmoid(CE logit) in [0, 1]."""
        if not candidates:
            return candidates
        try:
            if self.cross_encoder is None:
                from sentence_transformers import CrossEncoder
                print("Loading cross-encoder model (BAAI/bge-reranker-base)...")
                # BGE reranker handles cross-domain financial queries far better than the
                # ms-marco-MiniLM family, which scored "Provision for income taxes" passages
                # at 1.0 against revenue queries because of shared surface vocabulary.
                # device='cpu': WSL2's 2 GiB integrated GPU can't hold BGE + activations.
                self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base', device='cpu')

            pairs = [(query, c['chunk']) for c in candidates]
            raw_scores = self.cross_encoder.predict(pairs)

            for c, raw in zip(candidates, raw_scores):
                c['embedding_score'] = c.get('score')
                c['raw_ce_score'] = float(raw)
                # Sigmoid maps the CE logit to a calibrated [0, 1] relevance score.
                c['score'] = float(1.0 / (1.0 + np.exp(-float(raw))))
                c['method'] = c.get('method', 'embedding') + '_reranked'

            reranked = sorted(candidates, key=lambda x: x['score'], reverse=True)[:top_k]
            for i, c in enumerate(reranked):
                c['rank'] = i + 1

            top_score = reranked[0]['score'] if reranked else 0.0
            print(f"Cross-encoder re-rank done. Top-1 score: {top_score:.3f}")
            return reranked
        except Exception as e:
            print(f"Warning: cross-encoder re-rank failed ({e}); using embedding ranking.")
            return candidates[:top_k]

    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using the language model
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        # Instruction-style prompt: flan-t5 was trained to follow instructions, and an explicit
        # "if not in context, say so" clause stops it from copying an unrelated chunk verbatim.
        prompt = (
            "Read the financial context and answer the question concisely using only facts from the context. "
            "If the context does not contain the answer, reply exactly: "
            "\"The answer is not in the provided documents.\"\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        
        # flan-t5-large supports 1024 input tokens; give it room for more retrieved context.
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

        # max_new_tokens (not max_length) so the answer cap is independent of prompt length.
        # num_beams=2 (was 4) keeps most answer quality while halving inference memory —
        # critical on a 12 GB WSL2 budget where beam=4 OOM-killed the process.
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            min_length=30,
            num_beams=2,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    

    # Updated advanced_query method in the RAGSystem class      

    def advanced_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a query using advanced RAG techniques
        
        Args:
            query: User query
            k: Base number of chunks to retrieve
            
        Returns:
            Dictionary containing response and retrieval details
        """
        if not self.enable_advanced_retrieval or not hasattr(self, 'advanced_retrieval'):
            print("Advanced retrieval not available, falling back to basic retrieval")
            return self.query(query, k)
            
        # Initialize BM25 if needed
        if not hasattr(self.advanced_retrieval, 'bm25_index') or self.advanced_retrieval.bm25_index is None:
            self.advanced_retrieval.initialize_bm25(self.chunks)
        
        # Expand the query with financial terms
        expanded_query = self.expand_financial_query(query)

        # Use adaptive retrieval
        try:
            # Use the embedded adaptive_retrieval method
            retrieved_chunks, params = self.advanced_retrieval.adaptive_retrieval(
                query=query, 
                chunks=self.chunks,
                embeddings_fn=self.retrieve,
                k_values=[3, 5, 7]
            )
            
            # Always finish with cross-encoder re-rank so confidence reflects true relevance.
            # If adaptive_retrieval already reranked, this is a cheap re-score on the same chunks.
            retrieved_chunks = self._rerank_with_cross_encoder(query, retrieved_chunks, top_k=len(retrieved_chunks))

            # Combine chunks for context
            context = "\n".join([chunk['chunk'] for chunk in retrieved_chunks])

            # Generate response
            response = self.generate_response(query, context)

            # Confidence = top-1 cross-encoder score; no domain/method bonuses (they inflate scores
            # without evidence the answer is actually relevant).
            confidence_score = float(retrieved_chunks[0]['score']) if retrieved_chunks else 0.0

            print(f"Using method: Advanced (confidence: {confidence_score:.2f})")
            
            return {
                'response': response,
                'retrieved_chunks': retrieved_chunks,
                'confidence_score': confidence_score,
                'retrieval_params': params
            }
            
        except Exception as e:
            print(f"ERROR in advanced retrieval: {str(e)}. Falling back to basic retrieval")
            return self.query(query, k)


    def query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a query through the full RAG pipeline
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            Dictionary containing response and retrieval details
        """
        # Add logging
        print(f"Query: '{query}'")
        print(f"Advanced RAG enabled: {self.enable_advanced_retrieval}")
        print(f"Advanced RAG available: {hasattr(self, 'advanced_retrieval')}")
        
        if self.enable_advanced_retrieval and hasattr(self, 'advanced_retrieval'):
            return self.advanced_query(query, k)
            
        # Basic path: dense retrieval + always-on cross-encoder re-rank
        print("Using method: Basic + cross-encoder re-rank")

        # Over-fetch so the cross-encoder has more candidates to choose from
        retrieval_k = max(k * 2, 10)
        candidates = self.retrieve(query, retrieval_k)

        retrieved_chunks = self._rerank_with_cross_encoder(query, candidates, top_k=k)

        # Combine chunks for context
        context = "\n".join([chunk['chunk'] for chunk in retrieved_chunks])

        # Generate response
        response = self.generate_response(query, context)

        # Confidence = top-1 cross-encoder score (calibrated query-passage relevance)
        confidence = float(retrieved_chunks[0]['score']) if retrieved_chunks else 0.0

        return {
            'response': response,
            'retrieved_chunks': retrieved_chunks,
            'confidence_score': confidence,
        }
    
    # Add this new method to the RAGSystem class
    def expand_financial_query(self, query: str) -> str:
        """Expand financial queries with domain-specific synonyms"""
        financial_expansions = {
            'debt-to-equity': ['debt ratio', 'leverage ratio', 'D/E ratio', 'financial leverage'],
            'profit': ['earnings', 'net income', 'bottom line'],
            'revenue': ['sales', 'top line', 'income'],
            'assets': ['holdings', 'properties', 'resources'],
            'liabilities': ['debts', 'obligations', 'payables'],
            'equity': ['shareholder value', 'book value', 'net worth'],
            'margin': ['profitability', 'return', 'yield']
        }
        
        expanded_terms = []
        query_lower = query.lower()
        
        for term, expansions in financial_expansions.items():
            if term in query_lower:
                expanded_terms.extend(expansions)
        
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"
            print(f"Expanded query: '{expanded_query}'")
            return expanded_query
        return query

def main():
    # Initialize RAG system
    rag = RAGSystem(enable_advanced_retrieval=True)
    
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
        print(f"\nRank {chunk['rank']} (Score: {chunk['score']:.3f}, Method: {chunk.get('method', 'unknown')}):")
        print(chunk['chunk'][:300] + "..." if len(chunk['chunk']) > 300 else chunk['chunk'])
    
    # If advanced retrieval was used, print the parameters
    if 'retrieval_params' in result:
        print("\nRetrieval Parameters:")
        for k, v in result['retrieval_params'].items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()