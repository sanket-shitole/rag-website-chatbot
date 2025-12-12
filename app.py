"""
Streamlit application for RAG Website Chatbot.
"""

import os
import time
import logging
from typing import Optional
import streamlit as st
from dotenv import load_dotenv
from src.crawler import WebCrawler
from src.knowledge_base import KnowledgeBase
from src.rag_pipeline import RAGPipeline
from src.utils import is_valid_url, format_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Website Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1F77B4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1F77B4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #155A8A;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #D4EDDA;
        border: 1px solid #C3E6CB;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #F8D7DA;
        border: 1px solid #F5C6CB;
        color: #721C24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #D1ECF1;
        border: 1px solid #BEE5EB;
        color: #0C5460;
        margin: 1rem 0;
    }
    .source-card {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1F77B4;
        background-color: #F8F9FA;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'kb' not in st.session_state:
        st.session_state.kb = None
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'crawled_url' not in st.session_state:
        st.session_state.crawled_url = None
    if 'kb_stats' not in st.session_state:
        st.session_state.kb_stats = None


def validate_api_keys() -> tuple[Optional[str], Optional[str]]:
    """
    Validate and retrieve API keys.
    
    Returns:
        Tuple of (groq_api_key, google_api_key)
    """
    groq_key = os.getenv('GROQ_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')
    
    return groq_key, google_key


def crawl_and_build_kb(url: str, max_depth: int, max_pages: int, 
                       chunk_size: int, chunk_overlap: int):
    """
    Crawl website and build knowledge base.
    
    Args:
        url: Website URL to crawl
        max_depth: Maximum crawl depth
        max_pages: Maximum pages to crawl
        chunk_size: Text chunk size
        chunk_overlap: Chunk overlap size
    """
    try:
        # Create crawler
        crawler = WebCrawler(url, max_depth=max_depth, max_pages=max_pages)
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Start crawling
        status_text.text("🕷️ Starting web crawler...")
        progress_bar.progress(10)
        
        start_time = time.time()
        crawled_data = crawler.crawl()
        crawl_time = time.time() - start_time
        
        if not crawled_data:
            st.error("❌ No content was extracted from the website. Please check the URL and try again.")
            return False
        
        progress_bar.progress(40)
        status_text.text(f"✅ Crawled {len(crawled_data)} pages in {format_time(crawl_time)}")
        
        # Build knowledge base
        status_text.text("🔨 Building knowledge base...")
        progress_bar.progress(60)
        
        kb_start_time = time.time()
        kb = KnowledgeBase()
        kb.build_from_crawled_data(crawled_data, chunk_size=chunk_size, 
                                    chunk_overlap=chunk_overlap)
        kb_time = time.time() - kb_start_time
        
        progress_bar.progress(80)
        status_text.text("🔗 Initializing RAG pipeline...")
        
        # Initialize RAG pipeline
        groq_key, google_key = validate_api_keys()
        if not groq_key:
            st.error("❌ GROQ_API_KEY not found. Please set it in your .env file.")
            return False
        
        rag_pipeline = RAGPipeline(kb, groq_key, google_key)
        
        # Store in session state
        st.session_state.kb = kb
        st.session_state.rag_pipeline = rag_pipeline
        st.session_state.crawled_url = url
        st.session_state.kb_stats = kb.get_stats()
        
        progress_bar.progress(100)
        status_text.text("✅ Knowledge base ready!")
        
        # Show stats
        total_time = time.time() - start_time
        crawler_stats = crawler.get_stats()
        kb_stats = kb.get_stats()
        
        st.markdown(f"""
        <div class="success-box">
            <h4>✅ Knowledge Base Built Successfully!</h4>
            <ul>
                <li><b>Pages crawled:</b> {crawler_stats['pages_crawled']}</li>
                <li><b>Text chunks created:</b> {kb_stats['total_chunks']}</li>
                <li><b>Unique sources:</b> {kb_stats['unique_sources']}</li>
                <li><b>Crawl time:</b> {format_time(crawl_time)}</li>
                <li><b>KB build time:</b> {format_time(kb_time)}</li>
                <li><b>Total time:</b> {format_time(total_time)}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in crawl_and_build_kb: {str(e)}")
        st.error(f"❌ Error: {str(e)}")
        return False


def display_setup_page():
    """Display the setup page for crawling and KB creation."""
    st.markdown('<h1 class="main-header">🤖 RAG Website Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about any website using AI</p>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    ### How it works:
    1. **Enter a website URL** that you want to learn about
    2. **Configure crawling settings** (optional)
    3. **Click "Crawl & Build Knowledge Base"** to process the website
    4. **Ask questions** about the website content using natural language
    
    The chatbot will crawl the website, extract content, and use AI to answer your questions based on the information found.
    """)
    
    st.markdown("---")
    
    # URL input
    st.subheader("1️⃣ Enter Website URL")
    url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        help="Enter the URL of the website you want to learn about"
    )
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_depth = st.slider(
                "Max Crawl Depth",
                min_value=1,
                max_value=3,
                value=2,
                help="How many levels deep to crawl from the base URL"
            )
            
            max_pages = st.slider(
                "Max Pages",
                min_value=10,
                max_value=100,
                value=50,
                help="Maximum number of pages to crawl"
            )
        
        with col2:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=500,
                max_value=2000,
                value=1000,
                step=100,
                help="Size of text chunks for processing"
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=50,
                max_value=500,
                value=200,
                step=50,
                help="Overlap between consecutive chunks"
            )
    
    st.markdown("---")
    
    # Crawl button
    if st.button("🕷️ Crawl & Build Knowledge Base", type="primary"):
        if not url:
            st.error("❌ Please enter a website URL")
        elif not is_valid_url(url):
            st.error("❌ Please enter a valid URL (must start with http:// or https://)")
        else:
            with st.spinner("Processing website..."):
                success = crawl_and_build_kb(url, max_depth, max_pages, 
                                            chunk_size, chunk_overlap)
                if success:
                    st.success("✅ Ready to chat! Scroll down to start asking questions.")
                    st.rerun()


def display_chat_interface():
    """Display the chat interface."""
    st.markdown("### 💬 Chat Interface")
    
    # Display KB info
    if st.session_state.kb_stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Chunks", st.session_state.kb_stats['total_chunks'])
        with col2:
            st.metric("🔗 Sources", st.session_state.kb_stats['unique_sources'])
        with col3:
            st.metric("🌐 Website", st.session_state.crawled_url[:30] + "..." if len(st.session_state.crawled_url) > 30 else st.session_state.crawled_url)
    
    # Control buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset Knowledge Base"):
            st.session_state.kb = None
            st.session_state.rag_pipeline = None
            st.session_state.chat_history = []
            st.session_state.crawled_url = None
            st.session_state.kb_stats = None
            st.rerun()
    
    st.markdown("---")
    
    # Chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'user':
                st.markdown(f"**👤 You:** {content}")
            else:
                st.markdown(f"**🤖 Assistant:** {content}")
                
                # Display sources if available
                if 'sources' in message and message['sources']:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(message['sources'], 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <b>Source {i}:</b> {source.get('title', 'Untitled')}<br>
                                <a href="{source.get('url', '#')}" target="_blank">{source.get('url', '#')}</a><br>
                                <small>Relevance: {source.get('similarity_score', 0):.2%}</small>
                            </div>
                            """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    # Input box
    st.markdown("### Ask a Question")
    
    user_question = st.text_input(
        "Your question:",
        placeholder="What is this website about?",
        key="user_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        send_button = st.button("📤 Send", type="primary")
    
    if send_button and user_question:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question
        })
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.rag_pipeline.answer_question(
                    user_question,
                    chat_history=st.session_state.chat_history
                )
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response['answer'],
                    'sources': response.get('sources', [])
                })
                
                st.rerun()
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                st.error(f"❌ Error generating response: {str(e)}")


def main():
    """Main application function."""
    initialize_session_state()
    
    # Check API keys
    groq_key, google_key = validate_api_keys()
    if not groq_key:
        st.error("⚠️ GROQ_API_KEY not found in environment variables. Please create a .env file with your API key.")
        st.info("Get your Groq API key from: https://console.groq.com/")
        st.stop()
    
    # Display appropriate page
    if st.session_state.kb is None:
        display_setup_page()
    else:
        display_setup_page()
        st.markdown("---")
        display_chat_interface()


if __name__ == "__main__":
    main()
