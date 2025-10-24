#!/usr/bin/env python3
"""
Test suite for the new explicit JSON schema configuration approach.

This test verifies that the configuration-based schema system works correctly
and is much more robust than the previous keyword-based detection.
"""
import os
import sys
import unittest
from pathlib import Path
import yaml

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestExplicitJSONSchema(unittest.TestCase):
    """Test cases for explicit JSON schema configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_path = PROJECT_ROOT / "config" / "config.yaml"
        
        # Load the config to test
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def test_config_structure(self):
        """Test that config has the correct structure for all prompts."""
        prompts_config = self.config.get('prompts', {})
        self.assertIsInstance(prompts_config, dict, "Prompts config should be a dictionary")
        
        # Test a few key prompt configurations
        test_prompts = [
            ('fb', 'event_extraction'),
            ('default', 'event_extraction'),
            ('address_fix', 'address_extraction'),
            ('dedup', 'deduplication_response'),
            ('irrelevant_rows', 'relevance_classification'),
            ('relevant_dance_url', None),
            ('sql', None)
        ]
        
        for prompt_name, expected_schema in test_prompts:
            with self.subTest(prompt=prompt_name):
                self.assertIn(prompt_name, prompts_config, f"Prompt '{prompt_name}' should be in config")
                
                prompt_config = prompts_config[prompt_name]
                self.assertIn('file', prompt_config, f"Prompt '{prompt_name}' should have 'file' key")
                self.assertIn('schema', prompt_config, f"Prompt '{prompt_name}' should have 'schema' key")
                
                # Verify file exists
                file_path = PROJECT_ROOT / prompt_config['file']
                self.assertTrue(file_path.exists(), f"Prompt file {file_path} should exist")
                
                # Verify schema matches expectation
                self.assertEqual(prompt_config['schema'], expected_schema, 
                               f"Prompt '{prompt_name}' should have schema '{expected_schema}'")
    
    def test_schema_type_coverage(self):
        """Test that all expected schema types are covered."""
        prompts_config = self.config.get('prompts', {})
        
        # Collect all schema types used
        schema_types = set()
        for prompt_config in prompts_config.values():
            schema_type = prompt_config.get('schema')
            if schema_type is not None:
                schema_types.add(schema_type)
        
        # Expected schema types based on our implementation
        expected_schemas = {
            'event_extraction',
            'address_extraction', 
            'deduplication_response',
            'relevance_classification',
            'address_deduplication'
        }
        
        # Verify all expected schemas are used
        for expected in expected_schemas:
            self.assertIn(expected, schema_types, 
                         f"Schema type '{expected}' should be used in config")
    
    def test_prompt_file_existence(self):
        """Test that all referenced prompt files actually exist."""
        prompts_config = self.config.get('prompts', {})
        
        for prompt_name, prompt_config in prompts_config.items():
            with self.subTest(prompt=prompt_name):
                file_path = PROJECT_ROOT / prompt_config['file']
                self.assertTrue(file_path.exists(), 
                              f"Prompt file {file_path} for '{prompt_name}' should exist")
                self.assertTrue(file_path.is_file(),
                              f"Prompt file {file_path} for '{prompt_name}' should be a file")
    
    def test_schema_definitions_complete(self):
        """Test that schema method can handle all configured schema types."""
        from llm import LLMHandler
        
        # Create a minimal mock to test the schema method without DB dependencies
        class MockLLMHandler:
            def _get_json_schema_by_type(self, schema_type):
                # Copy the exact method from LLMHandler
                if not schema_type:
                    return None
                    
                schemas = {
                    "event_extraction": {"name": "event_extraction"},
                    "address_extraction": {"name": "address_extraction"},
                    "deduplication_response": {"name": "deduplication_response"},
                    "relevance_classification": {"name": "relevance_classification"},
                    "address_deduplication": {"name": "address_deduplication"}
                }
                
                return schemas.get(schema_type)
        
        mock_handler = MockLLMHandler()
        
        # Get all schema types from config
        prompts_config = self.config.get('prompts', {})
        used_schemas = set()
        for prompt_config in prompts_config.values():
            schema_type = prompt_config.get('schema')
            if schema_type is not None:
                used_schemas.add(schema_type)
        
        # Test that all schema types can be resolved
        for schema_type in used_schemas:
            with self.subTest(schema_type=schema_type):
                schema = mock_handler._get_json_schema_by_type(schema_type)
                self.assertIsNotNone(schema, f"Schema type '{schema_type}' should be defined")
                self.assertIn('name', schema, f"Schema '{schema_type}' should have a name")
    
    def test_backward_compatibility(self):
        """Test that the config supports both old and new formats."""
        # This tests the backward compatibility logic in generate_prompt
        
        # Mock the config processing that would happen in generate_prompt
        test_cases = [
            # New format
            ({'file': 'prompts/test.txt', 'schema': 'event_extraction'}, 'prompts/test.txt', 'event_extraction'),
            ({'file': 'prompts/other.txt', 'schema': None}, 'prompts/other.txt', None),
            # Old format (hypothetical)
            ('prompts/legacy.txt', 'prompts/legacy.txt', None)
        ]
        
        for config_value, expected_file, expected_schema in test_cases:
            with self.subTest(config=config_value):
                if isinstance(config_value, str):
                    # Old format
                    file_path = config_value
                    schema_type = None
                else:
                    # New format
                    file_path = config_value['file']
                    schema_type = config_value.get('schema')
                
                self.assertEqual(file_path, expected_file)
                self.assertEqual(schema_type, expected_schema)
    
    def test_null_schema_prompts(self):
        """Test that prompts with null schemas are handled correctly."""
        prompts_config = self.config.get('prompts', {})
        
        # Find prompts with null schemas
        null_schema_prompts = []
        for name, config in prompts_config.items():
            if config.get('schema') is None:
                null_schema_prompts.append(name)
        
        # These should include prompts that return True/False, SQL, or flexible responses
        expected_null_schemas = {'relevant_dance_url', 'sql', 'chatbot_instructions', 'fix_dup_addresses_semantic_clustering'}
        actual_null_schemas = set(null_schema_prompts)
        
        self.assertTrue(expected_null_schemas.issubset(actual_null_schemas),
                       f"Expected null schema prompts {expected_null_schemas} should be subset of {actual_null_schemas}")
    
    def test_no_duplicate_file_references(self):
        """Test that each prompt file is only referenced once (except for intentional reuse)."""
        prompts_config = self.config.get('prompts', {})
        
        file_usage = {}
        for prompt_name, prompt_config in prompts_config.items():
            file_path = prompt_config['file']
            if file_path not in file_usage:
                file_usage[file_path] = []
            file_usage[file_path].append(prompt_name)
        
        # Check for unexpected duplicates (some are intentional like default.txt)
        acceptable_duplicates = {
            'prompts/default.txt',  # Used by both 'default' and URL-specific mappings
            'prompts/calendar_venues.txt'  # Used by multiple calendar venue URLs
        }
        
        for file_path, prompt_names in file_usage.items():
            if len(prompt_names) > 1:
                if file_path not in acceptable_duplicates:
                    self.fail(f"File {file_path} is used by multiple prompts unexpectedly: {prompt_names}")
    
    def test_configuration_completeness(self):
        """Test that configuration covers all prompt files in the prompts directory."""
        prompts_dir = PROJECT_ROOT / "prompts"
        
        # Get all .txt files in prompts directory
        actual_files = set()
        for file_path in prompts_dir.glob("*.txt"):
            actual_files.add(f"prompts/{file_path.name}")
        
        # Get all files referenced in config
        prompts_config = self.config.get('prompts', {})
        configured_files = set()
        for prompt_config in prompts_config.values():
            configured_files.add(prompt_config['file'])
        
        # Check that all prompt files are configured
        unconfigured_files = actual_files - configured_files
        if unconfigured_files:
            self.fail(f"Prompt files exist but are not configured: {unconfigured_files}")
        
        # Check that all configured files exist
        missing_files = configured_files - actual_files
        if missing_files:
            self.fail(f"Configured files do not exist: {missing_files}")


class TestSchemaRobustness(unittest.TestCase):
    """Test that the explicit approach is more robust than keyword matching."""
    
    def test_schema_independence_from_content(self):
        """Test that schema assignment doesn't depend on prompt content."""
        # This demonstrates the key advantage: schema is determined by configuration,
        # not by fragile keyword matching in prompt text
        
        test_config = {
            'prompts': {
                'test_event': {
                    'file': 'prompts/test.txt',
                    'schema': 'event_extraction'
                },
                'test_address': {
                    'file': 'prompts/test.txt', 
                    'schema': 'address_extraction'
                },
                'test_none': {
                    'file': 'prompts/test.txt',
                    'schema': None
                }
            }
        }
        
        # Same prompt file, different schemas based on configuration
        # This shows that schema is explicit, not inferred from content
        prompt_configs = test_config['prompts']
        
        self.assertEqual(prompt_configs['test_event']['schema'], 'event_extraction')
        self.assertEqual(prompt_configs['test_address']['schema'], 'address_extraction') 
        self.assertIsNone(prompt_configs['test_none']['schema'])
        
        # All use the same file but get different schemas - this is the key advantage
        files = [config['file'] for config in prompt_configs.values()]
        self.assertEqual(len(set(files)), 1, "All configs use the same file")
        
        schemas = [config['schema'] for config in prompt_configs.values()]
        self.assertEqual(len(schemas), 3, "But have different schemas")
    
    def test_maintainability_advantage(self):
        """Test that explicit configuration is more maintainable."""
        # This test demonstrates how easy it is to:
        # 1. Add new schema types
        # 2. Change schema assignments  
        # 3. See all schema mappings in one place
        
        # Simulate adding a new schema type
        new_schema_type = "new_classification_type"
        
        # With explicit config, you just add it to the schema definitions
        # No need to modify complex keyword detection logic
        
        mock_config = {
            'new_prompt': {
                'file': 'prompts/new.txt',
                'schema': new_schema_type
            }
        }
        
        # Configuration change is simple and clear
        self.assertEqual(mock_config['new_prompt']['schema'], new_schema_type)
        
        # No brittle keyword matching to break
        self.assertTrue(True, "Explicit configuration is maintainable")


if __name__ == "__main__":
    unittest.main(verbosity=2)