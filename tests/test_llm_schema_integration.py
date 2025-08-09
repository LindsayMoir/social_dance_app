#!/usr/bin/env python3
"""
Integration tests for LLM schema system with real text processing.

These tests verify that the schema system works correctly without requiring 
actual LLM API calls (which would cost money and require API keys).
"""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestLLMSchemaIntegration(unittest.TestCase):
    """Integration tests for the LLM schema system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.maxDiff = None
        
        # Sample LLM responses that match our schemas
        self.sample_event_response = json.dumps([{
            "source": "Test Event Source",
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
        
        self.sample_address_response = json.dumps({
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
    
    def test_schema_by_type_definitions(self):
        """Test that _get_json_schema_by_type returns correct schemas."""
        # Import the method directly to test it
        sys.path.append(str(PROJECT_ROOT / "src"))
        from llm import LLMHandler
        
        # Create a mock handler to test the schema method
        handler = LLMHandler.__new__(LLMHandler)  # Create without __init__
        
        # Test event extraction schema
        schema = handler._get_json_schema_by_type('event_extraction')
        self.assertIsNotNone(schema)
        self.assertEqual(schema['name'], 'event_extraction')
        self.assertTrue(schema['strict'])
        self.assertEqual(schema['schema']['type'], 'array')
        self.assertIn('items', schema['schema'])
        
        # Verify required event fields
        properties = schema['schema']['items']['properties']
        required_fields = ['source', 'dance_style', 'url', 'event_type', 'event_name']
        for field in required_fields:
            self.assertIn(field, properties)
        
        # Test address extraction schema
        schema = handler._get_json_schema_by_type('address_extraction')
        self.assertIsNotNone(schema)
        self.assertEqual(schema['name'], 'address_extraction')
        self.assertEqual(schema['schema']['type'], 'object')
        
        # Verify required address fields
        properties = schema['schema']['properties']
        required_address_fields = ['address_id', 'full_address', 'street_number']
        for field in required_address_fields:
            self.assertIn(field, properties)
        
        # Test deduplication schema
        schema = handler._get_json_schema_by_type('deduplication_response')
        self.assertIsNotNone(schema)
        self.assertEqual(schema['name'], 'deduplication_response')
        self.assertEqual(schema['schema']['type'], 'array')
        
        dedup_properties = schema['schema']['items']['properties'] 
        self.assertIn('group_id', dedup_properties)
        self.assertIn('event_id', dedup_properties)
        self.assertIn('Label', dedup_properties)
        
        # Test None/nonexistent schema
        schema = handler._get_json_schema_by_type(None)
        self.assertIsNone(schema)
        
        schema = handler._get_json_schema_by_type('nonexistent')
        self.assertIsNone(schema)
    
    def test_query_method_schema_parameter_handling(self):
        """Test that query methods correctly handle schema_type parameter."""
        from llm import LLMHandler
        
        # Mock the external dependencies
        mock_openai_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = self.sample_event_response
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Create handler without full initialization
        handler = LLMHandler.__new__(LLMHandler)
        handler.openai_client = mock_openai_client
        
        # Test query_openai with schema
        result = handler.query_openai(
            prompt="Test prompt for event extraction",
            model="gpt-4",
            schema_type="event_extraction"
        )
        
        # Verify response
        self.assertEqual(result, self.sample_event_response)
        
        # Verify API was called with correct parameters
        call_args = mock_openai_client.chat.completions.create.call_args
        self.assertIsNotNone(call_args)
        
        # Check that response_format was included
        kwargs = call_args[1]  # Get keyword arguments
        self.assertIn('response_format', kwargs)
        response_format = kwargs['response_format']
        self.assertEqual(response_format['type'], 'json_schema')
        self.assertIn('json_schema', response_format)
        self.assertEqual(response_format['json_schema']['name'], 'event_extraction')
        
        # Test query_openai without schema
        mock_openai_client.reset_mock()
        result = handler.query_openai(
            prompt="Test prompt without schema",
            model="gpt-4"
            # No schema_type parameter
        )
        
        # Verify API was called without response_format
        call_args = mock_openai_client.chat.completions.create.call_args
        kwargs = call_args[1]
        self.assertNotIn('response_format', kwargs)
    
    def test_generate_prompt_config_parsing(self):
        """Test that generate_prompt correctly parses the new config format."""
        from llm import LLMHandler
        import tempfile
        import os
        
        # Create temporary prompt file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test prompt content for {event_type} events")
            temp_file = f.name
        
        try:
            # Mock config with new format
            mock_config = {
                'prompts': {
                    'test_with_schema': {
                        'file': temp_file,
                        'schema': 'event_extraction'
                    },
                    'test_no_schema': {
                        'file': temp_file,
                        'schema': None
                    },
                    'default': {
                        'file': temp_file,
                        'schema': 'event_extraction'
                    }
                }
            }
            
            # Create handler
            handler = LLMHandler.__new__(LLMHandler)
            handler.config = mock_config
            
            # Test new config format with schema
            prompt, schema_type = handler.generate_prompt(
                url="test://url",
                extracted_text="test event data",
                prompt_type="test_with_schema"
            )
            
            self.assertIn("Test prompt content", prompt)
            self.assertIn("test event data", prompt)
            self.assertEqual(schema_type, "event_extraction")
            
            # Test new config format without schema
            prompt, schema_type = handler.generate_prompt(
                url="test://url",
                extracted_text="test data",
                prompt_type="test_no_schema"
            )
            
            self.assertIn("Test prompt content", prompt)
            self.assertIsNone(schema_type)
            
            # Test fallback to default
            prompt, schema_type = handler.generate_prompt(
                url="test://url",
                extracted_text="test data",
                prompt_type="nonexistent_prompt"
            )
            
            self.assertIn("Test prompt content", prompt)
            self.assertEqual(schema_type, "event_extraction")  # Default schema
            
        finally:
            # Clean up temp file
            os.unlink(temp_file)
    
    def test_config_based_schema_assignment(self):
        """Test that schema assignment is based on config, not prompt content."""
        import yaml
        
        # Load actual config
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        prompts_config = config.get('prompts', {})
        
        # Verify key prompt-schema mappings
        expected_mappings = {
            'fb': 'event_extraction',
            'default': 'event_extraction', 
            'address_fix': 'address_extraction',
            'dedup': 'deduplication_response',
            'irrelevant_rows': 'relevance_classification',
            'relevant_dance_url': None,
            'sql': None
        }
        
        for prompt_name, expected_schema in expected_mappings.items():
            with self.subTest(prompt=prompt_name):
                self.assertIn(prompt_name, prompts_config)
                prompt_config = prompts_config[prompt_name]
                self.assertEqual(prompt_config['schema'], expected_schema)
    
    def test_json_response_structure_validation(self):
        """Test that our sample responses match the schema structure."""
        # Parse sample responses
        event_data = json.loads(self.sample_event_response)
        address_data = json.loads(self.sample_address_response)
        
        # Validate event response structure
        self.assertIsInstance(event_data, list)
        self.assertEqual(len(event_data), 1)
        
        event = event_data[0]
        required_event_fields = [
            'source', 'dance_style', 'url', 'event_type', 'event_name',
            'day_of_week', 'start_date', 'end_date', 'start_time', 'end_time',
            'price', 'location', 'description'
        ]
        
        for field in required_event_fields:
            self.assertIn(field, event, f"Event missing required field: {field}")
            self.assertIsInstance(event[field], str, f"Event field {field} should be string")
        
        # Validate address response structure
        self.assertIsInstance(address_data, dict)
        
        required_address_fields = [
            'address_id', 'full_address', 'street_number', 'street_name',
            'street_type', 'city', 'province_or_state', 'country_id'
        ]
        
        for field in required_address_fields:
            self.assertIn(field, address_data, f"Address missing required field: {field}")
        
        # Validate specific field types
        self.assertIsInstance(address_data['address_id'], int)
        self.assertIsInstance(address_data['full_address'], str)
    
    def test_schema_consistency_across_methods(self):
        """Test that all methods use consistent schema definitions."""
        from llm import LLMHandler
        
        handler = LLMHandler.__new__(LLMHandler)
        
        # Get schema for event extraction
        event_schema = handler._get_json_schema_by_type('event_extraction')
        
        # Verify schema structure consistency
        self.assertEqual(event_schema['name'], 'event_extraction')
        self.assertTrue(event_schema['strict'])
        
        # Check that the schema includes all expected properties
        properties = event_schema['schema']['items']['properties']
        required = event_schema['schema']['items']['required']
        
        # All required fields should be in properties
        for field in required:
            self.assertIn(field, properties, f"Required field {field} not in properties")
        
        # All properties should have type definitions
        for field, definition in properties.items():
            self.assertIn('type', definition, f"Property {field} missing type definition")
    
    def test_backward_compatibility_handling(self):
        """Test that the system handles both old and new config formats."""
        from llm import LLMHandler
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Legacy prompt format")
            temp_file = f.name
        
        try:
            # Test old config format (string instead of dict)
            old_config = {'prompts': {'legacy': temp_file}}
            
            handler = LLMHandler.__new__(LLMHandler)
            handler.config = old_config
            
            # Should handle old format gracefully
            prompt, schema_type = handler.generate_prompt(
                url="test://url",
                extracted_text="test",
                prompt_type="legacy"
            )
            
            self.assertIn("Legacy prompt format", prompt)
            self.assertIsNone(schema_type)  # Old format has no schema
            
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main(verbosity=2)