"""
RAG pipeline for question answering using retrieved context.
"""

import logging
from typing import List, Dict, Optional
from groq import Groq
import google.generativeai as genai
from src.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG pipeline that retrieves context and generates answers using LLM.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, groq_api_key: str, 
                 google_api_key: Optional[str] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            knowledge_base: Knowledge base for retrieval
            groq_api_key: Groq API key
            google_api_key: Google API key for fallback (optional)
        """
        self.knowledge_base = knowledge_base
        self.groq_api_key = groq_api_key
        self.google_api_key = google_api_key
        
        # Initialize Groq client
        self.groq_client = None
        if groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                logger.info("Groq client initialized")
            except Exception as e:
                logger.error(f"Error initializing Groq client: {str(e)}")
        
        # Initialize Google Gemini as fallback
        self.gemini_model = None
        if google_api_key:
            try:
                genai.configure(api_key=google_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                logger.info("Google Gemini client initialized as fallback")
            except Exception as e:
                logger.warning(f"Error initializing Gemini client: {str(e)}")
    
    def retrieve_context(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant context chunks for a question.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunk dictionaries
        """
        logger.info(f"Retrieving context for question: {question[:100]}...")
        
        results = self.knowledge_base.search(question, top_k=top_k)
        
        logger.info(f"Retrieved {len(results)} relevant chunks")
        
        return results
    
    def construct_prompt(self, question: str, context: List[Dict], 
                        history: Optional[List] = None) -> str:
        """
        Construct prompt for LLM with context and chat history.
        
        Args:
            question: User question
            context: Retrieved context chunks
            history: Chat history (optional)
            
        Returns:
            Formatted prompt string
        """
        # Format context
        context_text = ""
        for i, chunk in enumerate(context, 1):
            source_info = f"[Source {i}: {chunk.get('title', 'Untitled')} - {chunk.get('url', 'N/A')}]"
            context_text += f"\n\n{source_info}\n{chunk['text']}\n"
        
        # Format chat history if provided
        history_text = ""
        if history and len(history) > 0:
            history_text = "\n\nPrevious conversation:\n"
            for msg in history[-3:]:  # Include last 3 exchanges
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_text += f"{role.capitalize()}: {content}\n"
        
        # Construct final prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context from a website.

Context from website:
{context_text}
{history_text}

User Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context above
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be specific and cite which source(s) you're using when relevant
4. Keep your answer concise but complete
5. Use a friendly, conversational tone

Answer:"""
        
        return prompt
    
    def generate_answer_groq(self, prompt: str) -> str:
        """
        Generate answer using Groq API.
        
        Args:
            prompt: Prompt for LLM
            
        Returns:
            Generated answer
        """
        try:
            logger.info("Generating answer with Groq...")
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.1-70b-versatile",
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            logger.info("Answer generated successfully with Groq")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with Groq: {str(e)}")
            raise
    
    def generate_answer_gemini(self, prompt: str) -> str:
        """
        Generate answer using Google Gemini API.
        
        Args:
            prompt: Prompt for LLM
            
        Returns:
            Generated answer
        """
        try:
            logger.info("Generating answer with Gemini...")
            
            response = self.gemini_model.generate_content(prompt)
            answer = response.text
            
            logger.info("Answer generated successfully with Gemini")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {str(e)}")
            raise
    
    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer using available LLM (Groq with Gemini fallback).
        
        Args:
            prompt: Prompt for LLM
            
        Returns:
            Generated answer
        """
        # Try Groq first
        if self.groq_client:
            try:
                return self.generate_answer_groq(prompt)
            except Exception as e:
                logger.warning(f"Groq failed, trying Gemini fallback: {str(e)}")
        
        # Fallback to Gemini
        if self.gemini_model:
            try:
                return self.generate_answer_gemini(prompt)
            except Exception as e:
                logger.error(f"Gemini also failed: {str(e)}")
                raise Exception("Both Groq and Gemini APIs failed")
        
        raise Exception("No LLM API available. Please provide valid API keys.")
    
    def format_response(self, answer: str, sources: List[Dict]) -> Dict:
        """
        Format the final response with answer and sources.
        
        Args:
            answer: Generated answer
            sources: Source chunks used
            
        Returns:
            Formatted response dictionary
        """
        # Extract unique sources
        unique_sources = []
        seen_urls = set()
        
        for source in sources:
            url = source.get('url', '')
            if url and url not in seen_urls:
                unique_sources.append({
                    'url': url,
                    'title': source.get('title', 'Untitled'),
                    'similarity_score': source.get('similarity_score', 0)
                })
                seen_urls.add(url)
        
        return {
            'answer': answer,
            'sources': unique_sources,
            'num_sources': len(unique_sources)
        }
    
    def answer_question(self, question: str, chat_history: Optional[List] = None,
                       top_k: int = 5) -> Dict:
        """
        Answer a question using RAG pipeline.
        
        Args:
            question: User question
            chat_history: Previous chat messages (optional)
            top_k: Number of context chunks to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Processing question: {question[:100]}...")
        
        try:
            # Retrieve relevant context
            context = self.retrieve_context(question, top_k=top_k)
            
            if not context:
                return {
                    'answer': "I couldn't find any relevant information in the knowledge base to answer your question.",
                    'sources': [],
                    'num_sources': 0
                }
            
            # Construct prompt
            prompt = self.construct_prompt(question, context, chat_history)
            
            # Generate answer
            answer = self.generate_answer(prompt)
            
            # Format response
            response = self.format_response(answer, context)
            
            logger.info("Question answered successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'num_sources': 0,
                'error': str(e)
            }
