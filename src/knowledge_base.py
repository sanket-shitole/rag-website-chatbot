"""
Knowledge base construction using embeddings and FAISS vector store.
"""

import os
import pickle
import logging
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import clean_text

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Knowledge base for storing and retrieving document chunks using embeddings.
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the knowledge base.
        
        Args:
            embedding_model: Name of the sentence transformer model
        """
        self.embedding_model_name = embedding_model
        self.model = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.dimension = None
        
        # Load embedding model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode(["test"])
            self.dimension = test_embedding.shape[1]
            logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def chunk_text(self, text: str, metadata: Dict, 
                   chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split text
        chunks = text_splitter.split_text(text)
        
        # Create chunk dictionaries with metadata
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                'text': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                **metadata
            }
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32
            )
            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def build_vector_store(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Build FAISS vector store from chunks and embeddings.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: NumPy array of embeddings
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if len(chunks) == 0:
            raise ValueError("Cannot build vector store with no chunks")
        
        logger.info(f"Building FAISS index with {len(chunks)} chunks...")
        
        # Store chunks and metadata
        self.chunks = [chunk['text'] for chunk in chunks]
        self.chunk_metadata = chunks
        
        # Create FAISS index
        # Using IndexFlatL2 for exact search (cosine similarity via L2 on normalized vectors)
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to index
        self.index.add(embeddings_normalized.astype('float32'))
        
        logger.info(f"FAISS index built successfully. Total vectors: {self.index.ntotal}")
    
    def build_from_crawled_data(self, crawled_pages: List[Dict],
                                 chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Build knowledge base from crawled page data.
        
        Args:
            crawled_pages: List of crawled page dictionaries
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        if not crawled_pages:
            raise ValueError("No crawled pages provided")
        
        logger.info(f"Building knowledge base from {len(crawled_pages)} pages...")
        
        all_chunks = []
        
        # Process each page
        for page in crawled_pages:
            # Prepare metadata
            metadata = {
                'url': page.get('url', ''),
                'title': page.get('title', ''),
                'depth': page.get('depth', 0)
            }
            
            # Chunk the text
            text = page.get('text', '')
            if text:
                chunks = self.chunk_text(text, metadata, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
        
        if not all_chunks:
            raise ValueError("No text chunks created from crawled pages")
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(crawled_pages)} pages")
        
        # Extract text from chunks
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        
        # Create embeddings
        embeddings = self.create_embeddings(chunk_texts)
        
        # Build vector store
        self.build_vector_store(all_chunks, embeddings)
        
        logger.info("Knowledge base built successfully")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunk dictionaries with similarity scores
        """
        if not self.index or not self.chunks:
            logger.warning("Knowledge base not initialized")
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Normalize for cosine similarity
        query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        top_k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_embedding_normalized.astype('float32'), top_k)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunk_metadata):
                result = {
                    **self.chunk_metadata[idx],
                    'similarity_score': float(1 - distance),  # Convert distance to similarity
                    'rank': i + 1
                }
                results.append(result)
        
        return results
    
    def save(self, directory: str):
        """
        Save knowledge base to disk.
        
        Args:
            directory: Directory to save KB files
        """
        os.makedirs(directory, exist_ok=True)
        
        logger.info(f"Saving knowledge base to {directory}")
        
        # Save FAISS index
        index_path = os.path.join(directory, 'faiss_index.index')
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(directory, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'chunk_metadata': self.chunk_metadata,
                'embedding_model_name': self.embedding_model_name,
                'dimension': self.dimension
            }, f)
        
        logger.info("Knowledge base saved successfully")
    
    def load(self, directory: str):
        """
        Load knowledge base from disk.
        
        Args:
            directory: Directory containing KB files
        """
        logger.info(f"Loading knowledge base from {directory}")
        
        # Load FAISS index
        index_path = os.path.join(directory, 'faiss_index.index')
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = os.path.join(directory, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_metadata = data['chunk_metadata']
            self.dimension = data['dimension']
            
            # Reload model if different
            if data['embedding_model_name'] != self.embedding_model_name:
                self.embedding_model_name = data['embedding_model_name']
                self._load_model()
        
        logger.info(f"Knowledge base loaded successfully. {len(self.chunks)} chunks available")
    
    def get_stats(self) -> Dict:
        """
        Get knowledge base statistics.
        
        Returns:
            Dictionary with KB stats
        """
        if not self.chunks:
            return {
                'total_chunks': 0,
                'total_vectors': 0,
                'embedding_dimension': self.dimension or 0
            }
        
        unique_urls = set(metadata.get('url', '') for metadata in self.chunk_metadata)
        
        return {
            'total_chunks': len(self.chunks),
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.dimension,
            'unique_sources': len(unique_urls),
            'model_name': self.embedding_model_name
        }
