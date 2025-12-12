"""
Unit tests for the web crawler module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from src.crawler import WebCrawler
from src.utils import is_valid_url, normalize_url, clean_text


class TestURLValidation(unittest.TestCase):
    """Test URL validation functions."""
    
    def test_valid_urls(self):
        """Test that valid URLs are recognized."""
        valid_urls = [
            "https://www.example.com",
            "http://example.com",
            "https://example.com/path/to/page",
            "https://example.com/path?query=value"
        ]
        
        for url in valid_urls:
            self.assertTrue(is_valid_url(url), f"Should accept valid URL: {url}")
    
    def test_invalid_urls(self):
        """Test that invalid URLs are rejected."""
        invalid_urls = [
            "",
            "not a url",
            "ftp://example.com",
            "javascript:alert('xss')",
            None
        ]
        
        for url in invalid_urls:
            self.assertFalse(is_valid_url(url), f"Should reject invalid URL: {url}")
    
    def test_normalize_url(self):
        """Test URL normalization."""
        base = "https://example.com/page"
        
        # Test relative to absolute
        result = normalize_url("../other", base)
        self.assertEqual(result, "https://example.com/other")
        
        # Test fragment removal
        result = normalize_url("https://example.com/page#section")
        self.assertEqual(result, "https://example.com/page")


class TestTextCleaning(unittest.TestCase):
    """Test text cleaning functions."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        # Test HTML entity decoding
        text = "Hello &amp; goodbye"
        result = clean_text(text)
        self.assertEqual(result, "Hello & goodbye")
        
        # Test whitespace normalization
        text = "Hello    world\n\n\ntest"
        result = clean_text(text)
        self.assertEqual(result, "Hello world test")
        
        # Test empty string
        result = clean_text("")
        self.assertEqual(result, "")


class TestWebCrawler(unittest.TestCase):
    """Test WebCrawler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_url = "https://example.com"
        self.crawler = WebCrawler(self.base_url, max_depth=2, max_pages=10)
    
    def test_crawler_initialization(self):
        """Test crawler initializes correctly."""
        self.assertEqual(self.crawler.base_url, self.base_url)
        self.assertEqual(self.crawler.max_depth, 2)
        self.assertEqual(self.crawler.max_pages, 10)
        self.assertEqual(len(self.crawler.visited_urls), 0)
    
    def test_is_valid_url_for_crawling(self):
        """Test URL validation for crawling."""
        # Same domain - should be valid
        self.assertTrue(
            self.crawler.is_valid_url_for_crawling("https://example.com/page")
        )
        
        # Different domain - should be invalid
        self.assertFalse(
            self.crawler.is_valid_url_for_crawling("https://other.com/page")
        )
        
        # Invalid extension - should be invalid
        self.assertFalse(
            self.crawler.is_valid_url_for_crawling("https://example.com/image.jpg")
        )
    
    def test_extract_content(self):
        """Test content extraction from HTML."""
        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Heading</h1>
                <p>This is a paragraph with some content.</p>
                <script>console.log('ignore this');</script>
                <p>Another paragraph here.</p>
            </body>
        </html>
        """
        
        result = self.crawler.extract_content(html, self.base_url)
        
        self.assertEqual(result['title'], "Test Page")
        self.assertIn("Main Heading", result['headings'])
        self.assertIn("paragraph", result['text'].lower())
        self.assertNotIn("console.log", result['text'])
    
    def test_get_links(self):
        """Test link extraction from HTML."""
        html = """
        <html>
            <body>
                <a href="/page1">Page 1</a>
                <a href="https://example.com/page2">Page 2</a>
                <a href="https://other.com/page3">External</a>
                <a href="/image.jpg">Image</a>
            </body>
        </html>
        """
        
        links = self.crawler.get_links(html, self.base_url)
        
        # Should include same-domain HTML pages only
        urls = [link for link in links]
        self.assertTrue(any("page1" in url for url in urls))
        self.assertTrue(any("page2" in url for url in urls))
        # Should not include external domain or images
        self.assertFalse(any("other.com" in url for url in urls))
        self.assertFalse(any(".jpg" in url for url in urls))


class TestChunking(unittest.TestCase):
    """Test text chunking functionality."""
    
    def test_chunk_text(self):
        """Test that text is properly chunked."""
        from src.knowledge_base import KnowledgeBase
        
        kb = KnowledgeBase()
        
        # Create a long text
        text = "This is a sentence. " * 100  # 2000 chars
        metadata = {'url': 'https://example.com', 'title': 'Test'}
        
        chunks = kb.chunk_text(text, metadata, chunk_size=500, chunk_overlap=50)
        
        # Should create multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Each chunk should have metadata
        for chunk in chunks:
            self.assertIn('text', chunk)
            self.assertIn('url', chunk)
            self.assertIn('title', chunk)
            self.assertIn('chunk_index', chunk)


if __name__ == '__main__':
    unittest.main()
