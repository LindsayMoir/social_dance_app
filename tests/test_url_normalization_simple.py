#!/usr/bin/env python3
"""
Simple test for URL normalization functionality without full DatabaseHandler setup.
"""
import sys
import unittest
from pathlib import Path

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class SimpleUrlNormalizer:
    """Simplified version of URL normalization for testing."""
    
    def normalize_url(self, url):
        """
        Normalize URLs by removing dynamic cache parameters that don't affect the underlying content.
        """
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        
        parsed = urlparse(url)
        
        # Check if this is an Instagram or Facebook CDN URL
        instagram_domains = {
            'instagram.com',
            'www.instagram.com', 
            'scontent.cdninstagram.com',
            'instagram.fcxh2-1.fna.fbcdn.net',
            'scontent.cdninstagram.com'
        }
        
        fb_cdn_domains = {
            domain for domain in [parsed.netloc] 
            if 'fbcdn.net' in domain and ('instagram' in domain or 'scontent' in domain)
        }
        
        is_instagram_cdn = (parsed.netloc in instagram_domains or 
                           any(domain in parsed.netloc for domain in instagram_domains) or
                           bool(fb_cdn_domains))
        
        if not is_instagram_cdn:
            return url
        
        # Parse query parameters
        query_params = parse_qs(parsed.query)
        
        # List of dynamic parameters to remove for Instagram/FB CDN URLs
        dynamic_params = {
            '_nc_gid',     # Cache group ID - changes between sessions
            '_nc_ohc',     # Cache hash - changes between requests  
            '_nc_oc',      # Cache parameter - changes between requests
            'oh',          # Hash parameter - changes between requests
            'oe',          # Expiration parameter - changes over time
            '_nc_zt',      # Zoom/time parameter
            '_nc_ad',      # Ad parameter
            '_nc_cid',     # Cache ID
            'ccb',         # Cache control parameter (sometimes)
        }
        
        # Remove dynamic parameters
        filtered_params = {k: v for k, v in query_params.items() 
                          if k not in dynamic_params}
        
        # Reconstruct URL with filtered parameters
        new_query = urlencode(filtered_params, doseq=True)
        normalized_url = urlunparse((
            parsed.scheme,
            parsed.netloc, 
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
        
        return normalized_url


class TestUrlNormalizationSimple(unittest.TestCase):
    """Test URL normalization functionality."""
    
    def setUp(self):
        """Set up test."""
        self.normalizer = SimpleUrlNormalizer()
    
    def test_normalize_instagram_cdn_urls(self):
        """Test that Instagram CDN URLs with different cache parameters normalize to same URL."""
        
        # These are the actual URLs from the log that were being processed multiple times
        url1 = "https://scontent.cdninstagram.com/v/t51.71878-15/471612969_629684962743081_2572859140457891968_n.jpg?stp=dst-jpg_e15_tt6&_nc_cat=109&ig_cache_key=MzUzMjU3NDQwMzM1ODI2ODUwOQ%3D%3D.3-ccb1-7&ccb=1-7&_nc_sid=58cdad&efg=eyJ2ZW5jb2RlX3RhZyI6InhwaWRzLjY0MHgxMTM2LnNkciJ9&_nc_ohc=P0pmQfuoc-4Q7kNvwHDfgGD&_nc_oc=AdnuHsrlZ8A18mIOcjAJSUcKEtZtR1f74_zzYOQoRQpFmu7qHvsd-kiVyQiVwR_wk58&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent.cdninstagram.com&_nc_gid=8ZgOvAApXchHVfgva-Mjww&oh=00_AfNPVZdAUlERHghlOXek3LitXHVZ0AXUah1TH6np0R2XQA&oe=684EADC8"
        
        url2 = "https://scontent.cdninstagram.com/v/t51.71878-15/471612969_629684962743081_2572859140457891968_n.jpg?stp=dst-jpg_e15_tt6&_nc_cat=109&ig_cache_key=MzUzMjU3NDQwMzM1ODI2ODUwOQ%3D%3D.3-ccb1-7&ccb=1-7&_nc_sid=58cdad&efg=eyJ2ZW5jb2RlX3RhZyI6InhwaWRzLjY0MHgxMTM2LnNkciJ9&_nc_ohc=Ovz0t1hdFa4Q7kNvwEjRjs7&_nc_oc=AdnDvj51bCojRSkss_PqmPnckOojQr31q8chhMyZHJtNZ7nQ22i6xIupJDNW7mzqX6Q&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent.cdninstagram.com&_nc_gid=gGRTWulHHSFIifz8mm0J0Q&oh=00_AfNMZFgNHtXrY8JPhKy7JBkx0wIBiKHrlyQ9JLR0XfgL2A&oe=6850E048"
        
        url3 = "https://scontent.cdninstagram.com/v/t51.71878-15/471612969_629684962743081_2572859140457891968_n.jpg?stp=dst-jpg_e15_tt6&_nc_cat=109&ig_cache_key=MzUzMjU3NDQwMzM1ODI2ODUwOQ%3D%3D.3-ccb1-7&ccb=1-7&_nc_sid=58cdad&efg=eyJ2ZW5jb2RlX3RhZyI6InhwaWRzLjY0MHgxMTM2LnNkciJ9&_nc_ohc=P0pmQfuoc-4Q7kNvwHDfgGD&_nc_oc=AdnuHsrlZ8A18mIOcjAJSUcKEtZtR1f74_zzYOQoRQpFmu7qHvsd-kiVyQiVwR_wk58&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent.cdninstagram.com&_nc_gid=djnBP0CyfSZkJYHczlO2OQ&oh=00_AfMtixeV7ta0YGEHrfaPBgSgKkuCbmKRgwdH7tvwGZ1W8w&oe=684EADC8"
        
        # Normalize all three URLs
        normalized1 = self.normalizer.normalize_url(url1)
        normalized2 = self.normalizer.normalize_url(url2) 
        normalized3 = self.normalizer.normalize_url(url3)
        
        print(f"URL1 normalized to: {normalized1}")
        print(f"URL2 normalized to: {normalized2}")
        print(f"URL3 normalized to: {normalized3}")
        
        # All should normalize to the same URL
        self.assertEqual(normalized1, normalized2)
        self.assertEqual(normalized2, normalized3)
        self.assertEqual(normalized1, normalized3)
        
        # The normalized URL should not contain the dynamic parameters
        for param in ['_nc_gid', '_nc_ohc', '_nc_oc', 'oh', 'oe', '_nc_zt', '_nc_ad', '_nc_cid']:
            self.assertNotIn(param, normalized1, f"Parameter {param} should be removed")
            
        # But should still contain the essential parameters
        self.assertIn('stp=dst-jpg_e15_tt6', normalized1)
        self.assertIn('_nc_cat=109', normalized1)
        self.assertIn('ig_cache_key', normalized1)
    
    def test_normalize_non_instagram_urls_unchanged(self):
        """Test that non-Instagram URLs are not modified."""
        
        non_instagram_urls = [
            "https://example.com/image.jpg?param1=value1&param2=value2",
            "https://google.com/search?q=test&_nc_gid=something",  # Has Instagram param but different domain
            "https://facebook.com/photo?oh=test&oe=12345",  # Different domain
            "https://cdn.example.com/image.jpg?_nc_gid=test"  # Different domain
        ]
        
        for url in non_instagram_urls:
            normalized = self.normalizer.normalize_url(url)
            self.assertEqual(url, normalized, f"Non-Instagram URL should not be modified: {url}")


if __name__ == '__main__':
    unittest.main(verbosity=2)