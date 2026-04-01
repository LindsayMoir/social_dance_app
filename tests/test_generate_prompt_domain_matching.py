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
                    'file': self.domain_prompt_file,
                    'schema': 'event_extraction'
                },
                'https://loftpubvictoria.com/events/month/': {
                    'file': self.exact_url_prompt_file,
                    'schema': 'event_extraction'
                },
                # Exact URL config (existing behavior)
                'https://example.com/exact/path': {
                    'file': self.exact_url_prompt_file,
                    'schema': 'event_extraction'
                }
            }
        }
        
        # Create lightweight LLMHandler instance with mocked config.
        self.llm_handler = LLMHandler.__new__(LLMHandler)
        self.llm_handler.config = self.mock_config
        self.llm_handler._missing_prompt_types_logged = set()
    
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

                if prompt_type == "https://loftpubvictoria.com/events/month/":
                    self.assertIn("Exact URL prompt content", prompt)
                else:
                    self.assertIn("Domain-specific prompt content", prompt)
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
        
        # Should use domain-level prompt content rather than falling back to default.
        self.assertIn("Domain-specific prompt content", prompt)
        self.assertEqual(schema, 'event_extraction')

    def test_www_variant_matches_non_www_domain_config(self):
        """Test that www/non-www hostname differences still resolve the configured prompt."""
        self.mock_config["prompts"]["bardandbanker.com"] = {
            "file": self.domain_prompt_file,
            "schema": "event_extraction",
        }
        self.llm_handler = LLMHandler.__new__(LLMHandler)
        self.llm_handler.config = self.mock_config
        self.llm_handler._missing_prompt_types_logged = set()

        prompt, schema = self.llm_handler.generate_prompt(
            "https://www.bardandbanker.com/live-music",
            "Sample extracted text",
            "https://www.bardandbanker.com/live-music",
        )

        self.assertIn("Domain-specific prompt content", prompt)
        self.assertEqual(schema, "event_extraction")
    
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
        """Exact URL matches should avoid warnings and log the chosen prompt file."""
        url = "https://loftpubvictoria.com/some/page"
        prompt_type = "https://loftpubvictoria.com/events/month/"
        extracted_text = "Sample extracted text"
        
        self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
        
        info_messages = [call.args[0] for call in mock_info.call_args_list if call.args]
        self.assertTrue(
            any(
                "prompt type: https://loftpubvictoria.com/events/month/" in msg
                and self.exact_url_prompt_file in msg
                for msg in info_messages
            )
        )
        mock_warning.assert_not_called()
    
    @patch('logging.info')
    @patch('logging.warning')
    def test_unknown_url_domain_falls_back_to_default_without_warning(self, mock_warning, mock_info):
        """Unknown URL prompt types should fall back quietly because default prompts are expected."""
        url = "https://unknown-domain.com/some/page"
        prompt_type = "https://unknown-domain.com/some/page"
        extracted_text = "Sample extracted text"
        
        self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
        
        mock_warning.assert_not_called()
        mock_info.assert_any_call(
            "def generate_prompt(): Prompt type '%s' not found, using default",
            "https://unknown-domain.com/some/page",
        )

    @patch('logging.warning')
    @patch('logging.info')
    def test_unknown_url_domain_is_logged_once_per_domain(self, mock_info, mock_warning):
        """Repeated URL misses on the same domain should not spam fallback logs."""
        self.llm_handler.generate_prompt(
            "https://unknown-domain.com/one",
            "Sample extracted text",
            "https://unknown-domain.com/one",
        )
        self.llm_handler.generate_prompt(
            "https://unknown-domain.com/two",
            "Sample extracted text",
            "https://unknown-domain.com/two",
        )

        mock_warning.assert_not_called()
        mock_info.assert_any_call(
            "def generate_prompt(): Prompt type '%s' not found, using default",
            "https://unknown-domain.com/one",
        )

    @patch('logging.warning')
    def test_warning_for_unknown_non_url_prompt_type(self, mock_warning):
        """Non-URL prompt keys should still warn when they are misconfigured."""
        self.llm_handler.generate_prompt(
            "https://example.com/some/page",
            "Sample extracted text",
            "missing_prompt_key",
        )

        mock_warning.assert_called_once_with(
            "def generate_prompt(): Prompt type '%s' not found, using default",
            "missing_prompt_key",
        )


if __name__ == '__main__':
    unittest.main()
