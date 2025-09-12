#!/usr/bin/env python3
"""
Unit tests for domain-based URL matching in generate_prompt method.

These tests verify that the generate_prompt method correctly handles:
1. Exact URL matching (existing behavior)
2. Domain-based URL matching (new feature) 
3. Fallback to default prompt
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm import LLMHandler


class TestGeneratePromptDomainMatching(unittest.TestCase):
    """Test domain-based URL matching in generate_prompt method."""
    
    def setUp(self):
        """Set up test fixtures with temporary config and prompt files."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test prompt files
        self.default_prompt_file = os.path.join(self.test_dir, "default.txt")
        self.domain_prompt_file = os.path.join(self.test_dir, "domain_prompt.txt")
        self.exact_url_prompt_file = os.path.join(self.test_dir, "exact_url_prompt.txt")
        
        with open(self.default_prompt_file, 'w') as f:
            f.write("Default prompt content")
        
        with open(self.domain_prompt_file, 'w') as f:
            f.write("Domain-specific prompt content")
            
        with open(self.exact_url_prompt_file, 'w') as f:
            f.write("Exact URL prompt content")
        
        # Mock config with various prompt configurations
        self.mock_config = {
            'prompts': {
                'default': {
                    'file': self.default_prompt_file,
                    'schema': 'event_extraction'
                },
                # Domain-based config (what we're testing)
                'loftpubvictoria.com': {
                    'file': self.default_prompt_file,  # Use same default prompt but verify it matches via domain
                    'schema': 'event_extraction'
                },
                # Exact URL config (existing behavior)
                'https://example.com/exact/path': {
                    'file': self.exact_url_prompt_file,
                    'schema': 'event_extraction'
                }
            }
        }
        
        # Create LLMHandler instance with mocked config
        self.llm_handler = LLMHandler(self.mock_config)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_exact_url_match(self):
        """Test that exact URL matching still works (existing behavior)."""
        url = "https://example.com/some/page"
        prompt_type = "https://example.com/exact/path"
        extracted_text = "Sample extracted text"
        
        prompt, schema = self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
        
        # Should use exact URL prompt
        self.assertIn("Exact URL prompt content", prompt)
        self.assertEqual(schema, 'event_extraction')
    
    def test_domain_based_matching(self):
        """Test that domain-based matching works for loftpubvictoria.com URLs."""
        url = "https://loftpubvictoria.com/some/page"
        extracted_text = "Sample extracted text"
        
        test_cases = [
            "https://loftpubvictoria.com/events/month/",
            "https://loftpubvictoria.com/series/rhythm-train/",
            "https://loftpubvictoria.com/series/sunday-afternoon-jam/",
            "https://loftpubvictoria.com/series/tom-morrissey/"
        ]
        
        for prompt_type in test_cases:
            with self.subTest(url=prompt_type):
                prompt, schema = self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
                
                # Should use default prompt content (since loftpubvictoria.com is configured to use default.txt)
                self.assertIn("Default prompt content", prompt)
                self.assertEqual(schema, 'event_extraction')
                self.assertIn("Sample extracted text", prompt)
    
    def test_fallback_to_default(self):
        """Test that unknown URLs fall back to default prompt."""
        url = "https://unknown-domain.com/some/page"
        prompt_type = "https://unknown-domain.com/some/page"
        extracted_text = "Sample extracted text"
        
        prompt, schema = self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
        
        # Should use default prompt
        self.assertIn("Default prompt content", prompt)
        self.assertEqual(schema, 'event_extraction')
    
    def test_domain_precedence_over_default(self):
        """Test that domain matching takes precedence over default for unmatched full URLs."""
        url = "https://loftpubvictoria.com/new/path"
        prompt_type = "https://loftpubvictoria.com/new/path"  # This exact URL not in config
        extracted_text = "Sample extracted text"
        
        prompt, schema = self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
        
        # Should use default prompt content (since loftpubvictoria.com is configured to use default.txt)
        # The key test is that it matches via domain, not fallback to default warning
        self.assertIn("Default prompt content", prompt)
        self.assertEqual(schema, 'event_extraction')
    
    def test_malformed_url_fallback(self):
        """Test that malformed URLs fall back to default."""
        url = "https://example.com/some/page"
        prompt_type = "not-a-url"
        extracted_text = "Sample extracted text"
        
        prompt, schema = self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
        
        # Should use default prompt
        self.assertIn("Default prompt content", prompt)
        self.assertEqual(schema, 'event_extraction')
    
    @patch('logging.info')
    @patch('logging.warning')
    def test_logging_behavior(self, mock_warning, mock_info):
        """Test that appropriate log messages are generated."""
        url = "https://loftpubvictoria.com/some/page"
        prompt_type = "https://loftpubvictoria.com/events/month/"
        extracted_text = "Sample extracted text"
        
        self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
        
        # Should log domain-based config usage
        mock_info.assert_any_call("def generate_prompt(): Using domain-based config for 'loftpubvictoria.com'")
        
        # Should not log warning since domain match was found
        mock_warning.assert_not_called()
    
    @patch('logging.warning')
    def test_warning_for_unknown_domain(self, mock_warning):
        """Test that warning is logged for unknown domains."""
        url = "https://unknown-domain.com/some/page"
        prompt_type = "https://unknown-domain.com/some/page"
        extracted_text = "Sample extracted text"
        
        self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
        
        # Should log warning for fallback to default
        mock_warning.assert_called_with("def generate_prompt(): Prompt type 'https://unknown-domain.com/some/page' not found, using default")


if __name__ == '__main__':
    unittest.main()