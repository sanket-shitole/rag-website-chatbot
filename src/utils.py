"""
Utility functions for the RAG Website Chatbot.
"""

import re
import logging
from typing import Optional
from urllib.parse import urlparse, urljoin, urlunparse
import validators


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_valid_url(url: str) -> bool:
    """
    Validate if a URL is well-formed and has a valid scheme.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        if not url or not isinstance(url, str):
            return False
        
        # Use validators library for comprehensive validation
        result = validators.url(url)
        if not result:
            return False
            
        # Additional checks
        parsed = urlparse(url)
        if parsed.scheme not in ['http', 'https']:
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating URL {url}: {str(e)}")
        return False


def normalize_url(url: str, base_url: str = None) -> Optional[str]:
    """
    Normalize a URL by converting relative URLs to absolute and removing fragments.
    
    Args:
        url: URL to normalize
        base_url: Base URL for resolving relative URLs
        
    Returns:
        Normalized URL or None if invalid
    """
    try:
        if not url:
            return None
            
        # Convert relative URLs to absolute
        if base_url:
            url = urljoin(base_url, url)
            
        # Parse URL
        parsed = urlparse(url)
        
        # Remove fragment
        parsed = parsed._replace(fragment='')
        
        # Reconstruct URL
        normalized = urlunparse(parsed)
        
        return normalized if is_valid_url(normalized) else None
        
    except Exception as e:
        logger.error(f"Error normalizing URL {url}: {str(e)}")
        return None


def clean_text(text: str) -> str:
    """
    Clean and preprocess text by removing HTML entities, extra whitespace, etc.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove HTML entities
    import html
    text = html.unescape(text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special control characters but keep basic punctuation
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def get_domain(url: str) -> Optional[str]:
    """
    Extract domain from URL.
    
    Args:
        url: URL to extract domain from
        
    Returns:
        Domain string or None if invalid
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return None


def is_same_domain(url1: str, url2: str) -> bool:
    """
    Check if two URLs belong to the same domain.
    
    Args:
        url1: First URL
        url2: Second URL
        
    Returns:
        True if same domain, False otherwise
    """
    domain1 = get_domain(url1)
    domain2 = get_domain(url2)
    
    if not domain1 or not domain2:
        return False
        
    return domain1 == domain2


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
        
    return text[:max_length-3] + "..."
