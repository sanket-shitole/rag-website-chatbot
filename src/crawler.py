"""
Web crawler for extracting content from websites.
"""

import time
import logging
from typing import List, Dict, Set, Optional
from urllib.parse import urlparse, urljoin, urlunparse
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup
from src.utils import is_valid_url, normalize_url, is_same_domain, clean_text

logger = logging.getLogger(__name__)


class WebCrawler:
    """
    Web crawler that extracts content from websites with depth control.
    """
    
    def __init__(self, base_url: str, max_depth: int = 2, max_pages: int = 50):
        """
        Initialize the web crawler.
        
        Args:
            base_url: Starting URL for crawling
            max_depth: Maximum depth to crawl (default: 2)
            max_pages: Maximum number of pages to crawl (default: 50)
        """
        self.base_url = base_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.crawled_data: List[Dict] = []
        self.robot_parser = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; RAGBot/1.0; +https://github.com/sanket-shitole/rag-website-chatbot)'
        })
        
        # Initialize robots.txt parser
        self._init_robots_parser()
    
    def _init_robots_parser(self):
        """Initialize robots.txt parser for the domain."""
        try:
            parsed = urlparse(self.base_url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            self.robot_parser = RobotFileParser()
            self.robot_parser.set_url(robots_url)
            self.robot_parser.read()
            logger.info(f"Loaded robots.txt from {robots_url}")
        except Exception as e:
            logger.warning(f"Could not load robots.txt: {str(e)}")
            self.robot_parser = None
    
    def is_allowed_by_robots(self, url: str) -> bool:
        """
        Check if URL is allowed by robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            True if allowed or no robots.txt, False if disallowed
        """
        if not self.robot_parser:
            return True
            
        try:
            return self.robot_parser.can_fetch("*", url)
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
            return True
    
    def is_valid_url_for_crawling(self, url: str) -> bool:
        """
        Check if URL is valid for crawling.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL should be crawled, False otherwise
        """
        if not is_valid_url(url):
            return False
        
        # Check if same domain
        if not is_same_domain(url, self.base_url):
            return False
        
        # Check if already visited
        if url in self.visited_urls:
            return False
        
        # Check file extensions to skip
        skip_extensions = [
            '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
            '.css', '.js', '.json', '.xml', '.zip', '.tar', '.gz',
            '.mp4', '.mp3', '.avi', '.mov', '.doc', '.docx', '.xls',
            '.xlsx', '.ppt', '.pptx'
        ]
        
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        
        if any(path_lower.endswith(ext) for ext in skip_extensions):
            return False
        
        # Check robots.txt
        if not self.is_allowed_by_robots(url):
            logger.info(f"URL blocked by robots.txt: {url}")
            return False
        
        return True
    
    def extract_content(self, html: str, url: str) -> Dict:
        """
        Extract relevant content from HTML.
        
        Args:
            html: HTML content
            url: Source URL
            
        Returns:
            Dictionary with extracted content
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove script, style, and other non-content tags
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 
                         'aside', 'iframe', 'noscript', 'svg']):
            tag.decompose()
        
        # Extract title
        title = ""
        if soup.title:
            title = clean_text(soup.title.string or "")
        
        # Extract headings
        headings = []
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                text = clean_text(heading.get_text())
                if text:
                    headings.append(text)
        
        # Extract paragraphs and main text
        paragraphs = []
        for p in soup.find_all(['p', 'article', 'section', 'div']):
            text = clean_text(p.get_text())
            if text and len(text) > 20:  # Filter out very short text
                paragraphs.append(text)
        
        # Get all visible text
        visible_text = clean_text(soup.get_text())
        
        # Combine all text content
        all_text_parts = []
        if title:
            all_text_parts.append(f"Title: {title}")
        if headings:
            all_text_parts.append("Headings: " + " | ".join(headings))
        if paragraphs:
            all_text_parts.append("\n\n".join(paragraphs))
        
        combined_text = "\n\n".join(all_text_parts) if all_text_parts else visible_text
        
        return {
            'url': url,
            'title': title,
            'headings': headings,
            'paragraphs': paragraphs,
            'text': combined_text,
            'text_length': len(combined_text)
        }
    
    def get_links(self, html: str, current_url: str) -> List[str]:
        """
        Extract all valid links from HTML.
        
        Args:
            html: HTML content
            current_url: Current page URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        soup = BeautifulSoup(html, 'lxml')
        links = []
        
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            
            # Normalize URL
            absolute_url = normalize_url(href, current_url)
            
            if absolute_url and self.is_valid_url_for_crawling(absolute_url):
                links.append(absolute_url)
        
        return links
    
    def crawl_page(self, url: str, depth: int) -> Optional[Dict]:
        """
        Crawl a single page.
        
        Args:
            url: URL to crawl
            depth: Current depth level
            
        Returns:
            Extracted content or None if failed
        """
        try:
            logger.info(f"Crawling (depth {depth}): {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logger.info(f"Skipping non-HTML content: {url}")
                return None
            
            # Extract content
            content = self.extract_content(response.text, url)
            
            # Only add if has meaningful content
            if content['text_length'] > 100:
                content['depth'] = depth
                self.crawled_data.append(content)
                logger.info(f"Successfully extracted {content['text_length']} chars from {url}")
                
                # Get links if not at max depth
                if depth < self.max_depth:
                    links = self.get_links(response.text, url)
                    return {'content': content, 'links': links}
                
                return {'content': content, 'links': []}
            else:
                logger.info(f"Skipping page with insufficient content: {url}")
                return None
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout while crawling {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error crawling {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error crawling {url}: {str(e)}")
            return None
    
    def crawl(self) -> List[Dict]:
        """
        Start crawling from base URL.
        
        Returns:
            List of crawled page data with extracted content
        """
        logger.info(f"Starting crawl from {self.base_url}")
        logger.info(f"Max depth: {self.max_depth}, Max pages: {self.max_pages}")
        
        # Initialize queue with base URL at depth 0
        queue = [(self.base_url, 0)]
        
        while queue and len(self.crawled_data) < self.max_pages:
            # Get next URL from queue
            current_url, depth = queue.pop(0)
            
            # Skip if already visited
            if current_url in self.visited_urls:
                continue
            
            # Mark as visited
            self.visited_urls.add(current_url)
            
            # Crawl page
            result = self.crawl_page(current_url, depth)
            
            # Add new links to queue if within depth limit
            if result and result['links'] and depth < self.max_depth:
                for link in result['links']:
                    if link not in self.visited_urls and len(self.crawled_data) < self.max_pages:
                        queue.append((link, depth + 1))
            
            # Rate limiting - wait 1 second between requests
            time.sleep(1.0)
        
        logger.info(f"Crawl completed. Pages crawled: {len(self.crawled_data)}")
        
        return self.crawled_data
    
    def get_stats(self) -> Dict:
        """
        Get crawling statistics.
        
        Returns:
            Dictionary with crawl stats
        """
        total_text_length = sum(page['text_length'] for page in self.crawled_data)
        
        return {
            'pages_crawled': len(self.crawled_data),
            'urls_visited': len(self.visited_urls),
            'total_text_length': total_text_length,
            'average_text_length': total_text_length // len(self.crawled_data) if self.crawled_data else 0
        }
