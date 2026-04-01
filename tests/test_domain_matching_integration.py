#!/usr/bin/env python3
"""
Integration test for domain-based URL matching in generate_prompt method.

This test uses the actual config and prompt files to verify the domain matching works.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm import LLMHandler


class TestDomainMatchingIntegration(unittest.TestCase):
    """Integration test for domain matching using real config."""
    
    def setUp(self):
        """Set up test with real config."""
        # Load the actual config
        import yaml
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.llm_handler = LLMHandler.__new__(LLMHandler)
        self.llm_handler.config = self.config
        self.llm_handler._missing_prompt_types_logged = set()
    
    @patch('logging.info')
    @patch('logging.warning')
    def test_loftpubvictoria_domain_matching(self, mock_warning, mock_info):
        """Test that Loft listing/detail URLs resolve through the intended prompt split."""
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
                # Reset mocks for each test
                mock_info.reset_mock()
                mock_warning.reset_mock()
                
                prompt, schema = self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
                
                # Verify domain-based config was used (no warning logged)
                mock_warning.assert_not_called()
                
                # Verify the call stayed on the non-warning path.
                assert mock_info.call_count >= 1
                
                # Verify schema is correct
                self.assertEqual(schema, 'event_extraction')
                
                # Verify extracted text was included
                self.assertIn("Sample extracted text", prompt)

                if prompt_type == "https://loftpubvictoria.com/events/month/":
                    self.assertIn("multi-event venue listing pages", prompt)
                else:
                    self.assertIn("ONE EVENT ONLY", prompt)
    
    @patch('logging.info')
    @patch('logging.warning')
    def test_unknown_domain_falls_back_to_default(self, mock_warning, mock_info):
        """Test that unknown URL domains fall back quietly and use default."""
        url = "https://unknown-domain.com/some/page"
        prompt_type = "https://unknown-domain.com/some/page"
        extracted_text = "Sample extracted text"
        
        prompt, schema = self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
        
        mock_warning.assert_not_called()
        mock_info.assert_any_call(
            "def generate_prompt(): Prompt type '%s' not found, using default",
            "https://unknown-domain.com/some/page",
        )
        
        # Should still work
        self.assertEqual(schema, 'event_extraction')
        self.assertIn("Sample extracted text", prompt)
    
    def test_exact_url_matching_still_works(self):
        """Test that existing exact URL matching is not broken."""
        url = "https://gotothecoda.com/calendar"
        prompt_type = "https://gotothecoda.com/calendar"
        extracted_text = "Sample extracted text"
        
        prompt, schema = self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
        
        # Should use exact URL match (shared multi-event venue listing prompt)
        self.assertEqual(schema, 'event_extraction')
        self.assertIn("Sample extracted text", prompt)
        self.assertIn("multi-event venue listing pages", prompt)
        
    def test_config_has_loftpubvictoria_entry(self):
        """Test that the config actually has the Loft Pub Victoria prompt entry."""
        self.assertIn('https://loftpubvictoria.com/events/month/', self.config['prompts'])
        self.assertIn('loftpubvictoria.com', self.config['prompts'])

        loft_config = self.config['prompts']['https://loftpubvictoria.com/events/month/']
        self.assertEqual(loft_config['file'], 'prompts/multi_event_venues.txt')
        self.assertEqual(loft_config['schema'], 'event_extraction')

        loft_domain_config = self.config['prompts']['loftpubvictoria.com']
        self.assertEqual(loft_domain_config['file'], 'prompts/calendar_venues.txt')
        self.assertEqual(loft_domain_config['schema'], 'event_extraction')

    def test_multi_event_venue_prompt_entries_exist(self):
        """Listing-style venue URLs should use the shared multi-event venue prompt."""
        expected_keys = [
            'https://gotothecoda.com/calendar',
            'https://loftpubvictoria.com/events/month/',
            'https://thedukesaloon.com/events/',
            'https://www.irishtimespub.ca/live-music',
        ]
        for key in expected_keys:
            with self.subTest(key=key):
                self.assertIn(key, self.config['prompts'])
                self.assertEqual(self.config['prompts'][key]['file'], 'prompts/multi_event_venues.txt')


if __name__ == '__main__':
    unittest.main()
