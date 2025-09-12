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
        
        self.llm_handler = LLMHandler(self.config)
    
    @patch('logging.info')
    @patch('logging.warning')
    def test_loftpubvictoria_domain_matching(self, mock_warning, mock_info):
        """Test that loftpubvictoria.com URLs use domain-based matching."""
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
                
                # Verify domain-based matching log message was created
                mock_info.assert_any_call("def generate_prompt(): Using domain-based config for 'loftpubvictoria.com'")
                
                # Verify schema is correct
                self.assertEqual(schema, 'event_extraction')
                
                # Verify extracted text was included
                self.assertIn("Sample extracted text", prompt)
    
    @patch('logging.warning')
    def test_unknown_domain_falls_back_to_default(self, mock_warning):
        """Test that unknown domains still generate warning and use default."""
        url = "https://unknown-domain.com/some/page"
        prompt_type = "https://unknown-domain.com/some/page"
        extracted_text = "Sample extracted text"
        
        prompt, schema = self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
        
        # Should log warning for fallback to default
        mock_warning.assert_called_with("def generate_prompt(): Prompt type 'https://unknown-domain.com/some/page' not found, using default")
        
        # Should still work
        self.assertEqual(schema, 'event_extraction')
        self.assertIn("Sample extracted text", prompt)
    
    def test_exact_url_matching_still_works(self):
        """Test that existing exact URL matching is not broken."""
        url = "https://gotothecoda.com/calendar"
        prompt_type = "https://gotothecoda.com/calendar"
        extracted_text = "Sample extracted text"
        
        prompt, schema = self.llm_handler.generate_prompt(url, extracted_text, prompt_type)
        
        # Should use exact URL match (the_coda_prompt.txt)
        self.assertEqual(schema, 'event_extraction')
        self.assertIn("Sample extracted text", prompt)
        
    def test_config_has_loftpubvictoria_entry(self):
        """Test that the config actually has the loftpubvictoria.com entry."""
        self.assertIn('loftpubvictoria.com', self.config['prompts'])
        
        loft_config = self.config['prompts']['loftpubvictoria.com']
        self.assertEqual(loft_config['file'], 'prompts/default.txt')
        self.assertEqual(loft_config['schema'], 'event_extraction')


if __name__ == '__main__':
    unittest.main()