#!/usr/bin/env python3
"""
Integration tests for LLM query methods with JSON schema enforcement.

These tests verify that the updated query methods correctly apply JSON schemas
and return properly structured responses.
"""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestLLMIntegrationWithSchemas(unittest.TestCase):
    """Integration tests for LLM methods with schema enforcement."""
    
    def setUp(self):
        """Set up test fixtures with mocked LLM responses."""
        # We'll mock the actual API calls to avoid spending money during tests
        # but verify that the schema enforcement logic works correctly
        
        self.mock_event_response = json.dumps([{
            "source": "Test Source",
            "dance_style": "salsa, bachata",
            "url": "https://example.com/event",
            "event_type": "social dance",
            "event_name": "Latin Night",
            "day_of_week": "Friday",
            "start_date": "2025-08-15",
            "end_date": "2025-08-15", 
            "start_time": "20:00",
            "end_time": "23:00",
            "price": "$15",
            "location": "Dance Studio, 123 Main St",
            "description": "Fun Latin dance social"
        }])
        
        self.mock_address_response = json.dumps({
            "address_id": 0,
            "full_address": "Dance Studio, 123 Main St, Victoria, BC, V8V 1A1, CA",
            "building_name": "Dance Studio",
            "street_number": "123",
            "street_name": "Main",
            "street_type": "St",
            "direction": None,
            "city": "Victoria",
            "met_area": None,
            "province_or_state": "BC",
            "postal_code": "V8V 1A1",
            "country_id": "CA",
            "time_stamp": None
        })
        
        self.mock_dedup_response = json.dumps([
            {"group_id": 1, "event_id": 123, "Label": 0},
            {"group_id": 1, "event_id": 124, "Label": 1}
        ])
        
        self.mock_relevance_response = json.dumps([
            {"event_id": 123, "Label": 0, "event_type_new": "social dance"},
            {"event_id": 124, "Label": 1, "event_type_new": "other"}
        ])
        
        self.mock_address_dedup_response = json.dumps([
            {"address_id": 456, "Label": 0},
            {"address_id": 789, "Label": 1}
        ])
        
        self.mock_true_false_response = "True"
    
    @patch('openai.OpenAI')
    def test_event_extraction_with_schema(self):
        """Test event extraction with proper JSON schema enforcement."""
        from llm import LLMHandler
        
        # Mock OpenAI response
        mock_openai_instance = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = self.mock_event_response
        mock_openai_instance.chat.completions.create.return_value = mock_response
        
        with patch('openai.OpenAI', return_value=mock_openai_instance):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'test-key',
                'MISTRAL_API_KEY': 'test-key'
            }):
                # Create handler (will fail on DB init, but we can test the schema part)
                try:
                    handler = LLMHandler()
                    # If we get here, great! Otherwise we'll test the methods directly
                except:
                    # Create a minimal handler just for testing schema methods
                    handler = Mock()
                    handler._get_json_schema_by_type = LLMHandler._get_json_schema_by_type.__func__
                    handler.query_openai = LLMHandler.query_openai.__func__
                    handler.openai_client = mock_openai_instance
        
                # Test event extraction schema
                response = handler.query_openai(
                    handler,
                    prompt="Extract event details from: Latin dance night this Friday",
                    model="gpt-4",
                    schema_type="event_extraction"
                )
                
                # Verify the response
                self.assertEqual(response.strip(), self.mock_event_response)
                
                # Verify OpenAI was called with correct schema
                call_args = mock_openai_instance.chat.completions.create.call_args
                self.assertIn('response_format', call_args.kwargs)
                response_format = call_args.kwargs['response_format']
                self.assertEqual(response_format['type'], 'json_schema')
                self.assertIn('json_schema', response_format)
                self.assertEqual(response_format['json_schema']['name'], 'event_extraction')
    
    @patch('openai.OpenAI')
    def test_address_extraction_with_schema(self):
        """Test address extraction with proper JSON schema enforcement."""
        from llm import LLMHandler
        
        # Mock OpenAI response
        mock_openai_instance = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = self.mock_address_response
        mock_openai_instance.chat.completions.create.return_value = mock_response
        
        with patch('openai.OpenAI', return_value=mock_openai_instance):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'test-key',
                'MISTRAL_API_KEY': 'test-key'
            }):
                try:
                    handler = LLMHandler()
                except:
                    handler = Mock()
                    handler._get_json_schema_by_type = LLMHandler._get_json_schema_by_type.__func__
                    handler.query_openai = LLMHandler.query_openai.__func__
                    handler.openai_client = mock_openai_instance
        
                # Test address extraction schema
                response = handler.query_openai(
                    handler,
                    prompt="Extract address from: Dance Studio at 123 Main St, Victoria BC",
                    model="gpt-4",
                    schema_type="address_extraction"
                )
                
                # Verify the response
                self.assertEqual(response.strip(), self.mock_address_response)
                
                # Verify OpenAI was called with correct schema
                call_args = mock_openai_instance.chat.completions.create.call_args
                response_format = call_args.kwargs['response_format']
                self.assertEqual(response_format['json_schema']['name'], 'address_extraction')
    
    @patch('openai.OpenAI')
    def test_no_schema_enforcement(self):
        """Test that calls without schema_type don't enforce JSON schema."""
        from llm import LLMHandler
        
        # Mock OpenAI response
        mock_openai_instance = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = self.mock_true_false_response
        mock_openai_instance.chat.completions.create.return_value = mock_response
        
        with patch('openai.OpenAI', return_value=mock_openai_instance):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'test-key',
                'MISTRAL_API_KEY': 'test-key'
            }):
                try:
                    handler = LLMHandler()
                except:
                    handler = Mock()
                    handler._get_json_schema_by_type = LLMHandler._get_json_schema_by_type.__func__
                    handler.query_openai = LLMHandler.query_openai.__func__
                    handler.openai_client = mock_openai_instance
        
                # Test call without schema enforcement (e.g., True/False response)
                response = handler.query_openai(
                    handler,
                    prompt="Is this relevant for dance events?",
                    model="gpt-4"
                    # No schema_type parameter
                )
                
                # Verify the response
                self.assertEqual(response.strip(), self.mock_true_false_response)
                
                # Verify OpenAI was called WITHOUT response_format
                call_args = mock_openai_instance.chat.completions.create.call_args
                self.assertNotIn('response_format', call_args.kwargs)
    
    def test_schema_definitions_completeness(self):
        """Test that all schema types have proper definitions."""
        import yaml
        from pathlib import Path

        # Load config to verify schema types are defined
        config_path = Path("config/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Test all expected schema types have config entries
        schema_types = [
            'event_extraction',
            'address_extraction',
            'deduplication_response',
            'relevance_classification',
            'address_deduplication'
        ]

        # Verify these schema types are referenced in config prompts
        found_schemas = set()
        for prompt_name, prompt_config in config.get('prompts', {}).items():
            schema = prompt_config.get('schema')
            if schema and schema in schema_types:
                found_schemas.add(schema)

        # All schema types should be defined in at least one prompt config
        for schema_type in schema_types:
            with self.subTest(schema_type=schema_type):
                self.assertIn(schema_type, found_schemas,
                             f"Schema type '{schema_type}' should be defined in config prompts")
    
    def test_schema_type_none_handling(self):
        """Test that null schema types are handled correctly in config."""
        import yaml
        from pathlib import Path

        # Load config to verify null schema handling
        config_path = Path("config/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Count prompts with null schema
        null_schema_count = 0
        for prompt_name, prompt_config in config.get('prompts', {}).items():
            if prompt_config.get('schema') is None:
                null_schema_count += 1

        # There should be some prompts with null schema (like chatbot_instructions)
        self.assertGreater(null_schema_count, 0,
                          "There should be prompts with null schema for non-extraction tasks")
    
    def test_generate_prompt_with_schema_config(self):
        """Test that generate_prompt correctly extracts schema from config."""
        import yaml
        from llm import LLMHandler
        
        # Mock config structure
        mock_config = {
            'prompts': {
                'test_event': {
                    'file': 'prompts/fb_prompt.txt',
                    'schema': 'event_extraction'
                },
                'test_none': {
                    'file': 'prompts/relevant_dance_url.txt',
                    'schema': None
                },
                'test_legacy': 'prompts/old_format.txt'  # Legacy format
            }
        }
        
        # Create handler with mocked config
        handler = Mock()
        handler.config = mock_config
        handler.generate_prompt = LLMHandler.generate_prompt.__func__
        
        # Mock file reading
        with patch('builtins.open', unittest.mock.mock_open(read_data="Test prompt content")):
            # Test new format with schema
            prompt, schema_type = handler.generate_prompt(
                handler, 
                url="test_url",
                extracted_text="test text",
                prompt_type="test_event"
            )
            
            self.assertIn("Test prompt content", prompt)
            self.assertEqual(schema_type, "event_extraction")
            
            # Test new format with None schema
            prompt, schema_type = handler.generate_prompt(
                handler,
                url="test_url", 
                extracted_text="test text",
                prompt_type="test_none"
            )
            
            self.assertIn("Test prompt content", prompt)
            self.assertIsNone(schema_type)
            
            # Test legacy format
            prompt, schema_type = handler.generate_prompt(
                handler,
                url="test_url",
                extracted_text="test text", 
                prompt_type="test_legacy"
            )
            
            self.assertIn("Test prompt content", prompt)
            self.assertIsNone(schema_type)
    
    @patch('mistralai.Mistral')
    def test_mistral_integration_with_schema(self):
        """Test that Mistral integration also works with schemas."""
        from llm import LLMHandler
        
        # Mock Mistral response
        mock_mistral_instance = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = self.mock_event_response
        mock_mistral_instance.chat.complete.return_value = mock_response
        
        with patch('mistralai.Mistral', return_value=mock_mistral_instance):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'test-key',
                'MISTRAL_API_KEY': 'test-key'
            }):
                try:
                    handler = LLMHandler()
                except:
                    handler = Mock()
                    handler._get_json_schema_by_type = LLMHandler._get_json_schema_by_type.__func__
                    handler.query_mistral = LLMHandler.query_mistral.__func__
                    handler.mistral_client = mock_mistral_instance
        
                # Test Mistral with schema
                response = handler.query_mistral(
                    handler,
                    prompt="Extract event details from: Latin dance night this Friday",
                    model="mistral-large-2411",
                    schema_type="event_extraction"
                )
                
                # Verify the response
                self.assertEqual(response.strip(), self.mock_event_response)
                
                # Verify Mistral was called with correct schema
                call_args = mock_mistral_instance.chat.complete.call_args
                self.assertIn('response_format', call_args.kwargs)
                response_format = call_args.kwargs['response_format']
                self.assertEqual(response_format['type'], 'json_schema')
                self.assertEqual(response_format['json_schema']['name'], 'event_extraction')
    
    def test_json_response_validation(self):
        """Test that returned JSON responses match expected schema structure."""
        # Test that our mock responses would actually validate against the schemas
        
        # Test event extraction response
        event_data = json.loads(self.mock_event_response)
        self.assertIsInstance(event_data, list)
        self.assertTrue(len(event_data) > 0)
        
        event = event_data[0]
        required_fields = [
            "source", "dance_style", "url", "event_type", "event_name",
            "day_of_week", "start_date", "end_date", "start_time", "end_time",
            "price", "location", "description"
        ]
        
        for field in required_fields:
            self.assertIn(field, event, f"Event should have {field} field")
            self.assertIsInstance(event[field], str, f"{field} should be string")
        
        # Test address extraction response
        address_data = json.loads(self.mock_address_response)
        self.assertIsInstance(address_data, dict)
        
        required_address_fields = [
            "address_id", "full_address", "street_number", "street_name",
            "street_type", "city", "province_or_state", "country_id"
        ]
        
        for field in required_address_fields:
            self.assertIn(field, address_data, f"Address should have {field} field")
        
        # Test deduplication response
        dedup_data = json.loads(self.mock_dedup_response)
        self.assertIsInstance(dedup_data, list)
        
        for item in dedup_data:
            self.assertIn("group_id", item)
            self.assertIn("event_id", item)
            self.assertIn("Label", item)
            self.assertIsInstance(item["Label"], int)
            self.assertIn(item["Label"], [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)