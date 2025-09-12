#!/usr/bin/env python3
"""
Test for Instagram URL normalization functionality.

Tests that duplicate Instagram URLs with different cache parameters
are properly normalized to prevent duplicate processing.
"""
import sys
import unittest
from pathlib import Path

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from db import DatabaseHandler


class TestInstagramUrlNormalization(unittest.TestCase):
    """Test Instagram URL normalization to prevent duplicate processing."""
    
    def setUp(self):
        """Set up test with mock config."""
        # Create minimal config for DatabaseHandler
        self.mock_config = {
            'testing': {
                'use_db': False  # Don't actually connect to database
            }
        }
        
        # Initialize database handler (without actual DB connection)
        self.db_handler = DatabaseHandler(self.mock_config)
    
    def test_normalize_instagram_cdn_urls(self):
        """Test that Instagram CDN URLs with different cache parameters normalize to same URL."""
        
        # These are the actual URLs from the log that were being processed multiple times
        url1 = "https://scontent.cdninstagram.com/v/t51.71878-15/471612969_629684962743081_2572859140457891968_n.jpg?stp=dst-jpg_e15_tt6&_nc_cat=109&ig_cache_key=MzUzMjU3NDQwMzM1ODI2ODUwOQ%3D%3D.3-ccb1-7&ccb=1-7&_nc_sid=58cdad&efg=eyJ2ZW5jb2RlX3RhZyI6InhwaWRzLjY0MHgxMTM2LnNkciJ9&_nc_ohc=P0pmQfuoc-4Q7kNvwHDfgGD&_nc_oc=AdnuHsrlZ8A18mIOcjAJSUcKEtZtR1f74_zzYOQoRQpFmu7qHvsd-kiVyQiVwR_wk58&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent.cdninstagram.com&_nc_gid=8ZgOvAApXchHVfgva-Mjww&oh=00_AfNPVZdAUlERHghlOXek3LitXHVZ0AXUah1TH6np0R2XQA&oe=684EADC8"
        
        url2 = "https://scontent.cdninstagram.com/v/t51.71878-15/471612969_629684962743081_2572859140457891968_n.jpg?stp=dst-jpg_e15_tt6&_nc_cat=109&ig_cache_key=MzUzMjU3NDQwMzM1ODI2ODUwOQ%3D%3D.3-ccb1-7&ccb=1-7&_nc_sid=58cdad&efg=eyJ2ZW5jb2RlX3RhZyI6InhwaWRzLjY0MHgxMTM2LnNkciJ9&_nc_ohc=Ovz0t1hdFa4Q7kNvwEjRjs7&_nc_oc=AdnDvj51bCojRSkss_PqmPnckOojQr31q8chhMyZHJtNZ7nQ22i6xIupJDNW7mzqX6Q&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent.cdninstagram.com&_nc_gid=gGRTWulHHSFIifz8mm0J0Q&oh=00_AfNMZFgNHtXrY8JPhKy7JBkx0wIBiKHrlyQ9JLR0XfgL2A&oe=6850E048"
        
        url3 = "https://scontent.cdninstagram.com/v/t51.71878-15/471612969_629684962743081_2572859140457891968_n.jpg?stp=dst-jpg_e15_tt6&_nc_cat=109&ig_cache_key=MzUzMjU3NDQwMzM1ODI2ODUwOQ%3D%3D.3-ccb1-7&ccb=1-7&_nc_sid=58cdad&efg=eyJ2ZW5jb2RlX3RhZyI6InhwaWRzLjY0MHgxMTM2LnNkciJ9&_nc_ohc=P0pmQfuoc-4Q7kNvwHDfgGD&_nc_oc=AdnuHsrlZ8A18mIOcjAJSUcKEtZtR1f74_zzYOQoRQpFmu7qHvsd-kiVyQiVwR_wk58&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent.cdninstagram.com&_nc_gid=djnBP0CyfSZkJYHczlO2OQ&oh=00_AfMtixeV7ta0YGEHrfaPBgSgKkuCbmKRgwdH7tvwGZ1W8w&oe=684EADC8"
        
        # Normalize all three URLs
        normalized1 = self.db_handler.normalize_url(url1)
        normalized2 = self.db_handler.normalize_url(url2) 
        normalized3 = self.db_handler.normalize_url(url3)
        
        # All should normalize to the same URL
        self.assertEqual(normalized1, normalized2)
        self.assertEqual(normalized2, normalized3)
        self.assertEqual(normalized1, normalized3)
        
        # The normalized URL should not contain the dynamic parameters
        for param in ['_nc_gid', '_nc_ohc', 'oh', 'oe', '_nc_zt', '_nc_ad', '_nc_cid']:
            self.assertNotIn(param, normalized1)
            
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
            normalized = self.db_handler.normalize_url(url)
            self.assertEqual(url, normalized, f"Non-Instagram URL should not be modified: {url}")
    
    def test_normalize_instagram_main_site_urls_unchanged(self):
        """Test that main Instagram site URLs (not CDN) are not modified."""
        
        instagram_main_urls = [
            "https://www.instagram.com/bachatavictoria/",
            "https://instagram.com/alivetangovictoria/?hl=en",
            "https://www.instagram.com/p/DNjbJK4ycki/"
        ]
        
        for url in instagram_main_urls:
            normalized = self.db_handler.normalize_url(url)
            # These should remain unchanged as they're not CDN URLs
            self.assertEqual(url, normalized, f"Instagram main site URL should not be modified: {url}")
    
    def test_normalize_instagram_fbcdn_urls(self):
        """Test that Instagram FBCDN URLs are properly normalized."""
        
        fbcdn_url1 = "https://instagram.fcxh2-1.fna.fbcdn.net/v/t51.2885-15/543639719_18527375536010389_4050783906703240306_n.jpg?stp=dst-jpg_e35_s640x640_sh0.08_tt6&_nc_ht=instagram.fcxh2-1.fna.fbcdn.net&_nc_cat=101&_nc_oc=Q6cZ2QEZX1ZAmDNT-cP512gfbI2Kyac-VK5M0QU2OcAPs7IwtNsJ3F1GfYPk32LbLuA7n68&_nc_ohc=LreyzYDrF7wQ7kNvwFoszH4&_nc_gid=5k1OjNjjJv9I5F8bbCgftQ&edm=AOQ1c0wBAAAA&ccb=7-5&oh=00_AfaPCr8dCw4yy4kuP2BIS7nefh3YyEsNFV3yIvjiXvoUqQ&oe=68C81FBC&_nc_sid=8b3546"
        
        fbcdn_url2 = "https://instagram.fcxh2-1.fna.fbcdn.net/v/t51.2885-15/543639719_18527375536010389_4050783906703240306_n.jpg?stp=dst-jpg_e35_s640x640_sh0.08_tt6&_nc_ht=instagram.fcxh2-1.fna.fbcdn.net&_nc_cat=101&_nc_oc=Q6cZ2QEZX1ZAmDNT-cP512gfbI2Kyac-VK5M0QU2OcAPs7IwtNsJ3F1GfYPk32LbLuA7n68&_nc_ohc=DifferentHash&_nc_gid=DifferentGid&edm=AOQ1c0wBAAAA&ccb=7-5&oh=00_DifferentOhValue&oe=DifferentOe&_nc_sid=8b3546"
        
        normalized1 = self.db_handler.normalize_url(fbcdn_url1)
        normalized2 = self.db_handler.normalize_url(fbcdn_url2)
        
        # Should normalize to same URL
        self.assertEqual(normalized1, normalized2)
        
        # Should remove dynamic parameters
        for param in ['_nc_gid', '_nc_ohc', 'oh', 'oe']:
            self.assertNotIn(param, normalized1)
    
    def test_edge_cases(self):
        """Test edge cases for URL normalization."""
        
        # URL with no query parameters
        simple_url = "https://scontent.cdninstagram.com/image.jpg"
        self.assertEqual(self.db_handler.normalize_url(simple_url), simple_url)
        
        # URL with only non-dynamic parameters
        static_url = "https://scontent.cdninstagram.com/image.jpg?stp=dst-jpg_e15_tt6&_nc_cat=109"
        self.assertEqual(self.db_handler.normalize_url(static_url), static_url)
        
        # Empty URL
        self.assertEqual(self.db_handler.normalize_url(""), "")


if __name__ == '__main__':
    unittest.main()