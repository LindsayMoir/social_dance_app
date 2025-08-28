#!/usr/bin/env python3
"""
Comprehensive tests for LLM schema handling and response parsing.
Tests both Mistral and OpenAI schema formats and parsing logic.
"""

import json
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm import LLMHandler
from unittest.mock import Mock, patch, mock_open
import yaml


class TestLLMSchemaHandling:
    """Test LLM schema generation and response parsing"""
    
    def setup_method(self):
        """Setup test configuration and LLM handler"""
        self.config = {
            'llm': {
                'provider': 'mistral',
                'spend_money': True,
                'mistral_model': 'mistral-large-latest',
                'openai_model': 'gpt-4o-mini'
            },
            'crawling': {
                'prompt_max_length': 50000
            },
            'input': {
                'data_keywords': 'test_keywords.csv',
                'fb_urls': 'test_fb_urls.csv',
                'gs_urls': 'test_gs_urls.csv',
                'emails': 'test_emails.csv',
                'edge_cases': 'test_edge_cases.csv',
                'urls': 'test_urls'
            },
            'output': {
                'events': 'test_events.csv'
            },
            'prompts': {
                'default': 'prompts/default.txt'
            }
        }
        
        # Mock clients and config loading to avoid actual API calls and file reads
        with patch('llm.Mistral'), patch('llm.OpenAI'), \
             patch('llm.DatabaseHandler'), \
             patch('builtins.open', mock_open()), \
             patch('llm.yaml.safe_load', return_value=self.config), \
             patch('llm.os.getenv', return_value='test_key'), \
             patch('llm.pd.read_csv', return_value=Mock(columns=['keyword'], to_list=Mock(return_value=['dance', 'swing']))):
            self.llm_handler = LLMHandler()
            

    def test_mistral_event_extraction_schema(self):
        """Test Mistral gets correct event extraction schema (array format)"""
        schema = self.llm_handler._get_json_schema_by_type("event_extraction", "mistral")
        
        assert schema is not None
        assert schema["name"] == "event_extraction"
        assert schema["strict"] is True
        assert schema["schema"]["type"] == "array"
        assert "items" in schema["schema"]
        assert schema["schema"]["items"]["type"] == "object"
        
        # Check required event properties
        props = schema["schema"]["items"]["properties"]
        required = schema["schema"]["items"]["required"]
        
        expected_props = ["source", "dance_style", "url", "event_type", "event_name", 
                         "day_of_week", "start_date", "end_date", "start_time", "end_time", 
                         "price", "location", "description"]
        
        for prop in expected_props:
            assert prop in props
            assert prop in required

    def test_openai_event_extraction_schema(self):
        """Test OpenAI gets correct event extraction schema (wrapped object format)"""
        schema = self.llm_handler._get_json_schema_by_type("event_extraction", "openai")
        
        assert schema is not None
        assert schema["name"] == "event_extraction"
        assert schema["strict"] is True
        assert schema["schema"]["type"] == "object"
        assert "events" in schema["schema"]["properties"]
        assert schema["schema"]["properties"]["events"]["type"] == "array"
        
        # Check nested event structure
        event_items = schema["schema"]["properties"]["events"]["items"]
        props = event_items["properties"]
        required = event_items["required"]
        
        expected_props = ["source", "dance_style", "url", "event_type", "event_name", 
                         "day_of_week", "start_date", "end_date", "start_time", "end_time", 
                         "price", "location", "description"]
        
        for prop in expected_props:
            assert prop in props
            assert prop in required

    def test_address_extraction_schemas_identical(self):
        """Test that address extraction schemas are identical for both providers"""
        mistral_schema = self.llm_handler._get_json_schema_by_type("address_extraction", "mistral")
        openai_schema = self.llm_handler._get_json_schema_by_type("address_extraction", "openai")
        
        # Both should be object type (not wrapped)
        assert mistral_schema["schema"]["type"] == "object"
        assert openai_schema["schema"]["type"] == "object"
        
        # Properties should be identical
        assert mistral_schema["schema"]["properties"] == openai_schema["schema"]["properties"]
        assert mistral_schema["schema"]["required"] == openai_schema["schema"]["required"]

    def test_deduplication_schemas_different(self):
        """Test that deduplication schemas are different for providers"""
        mistral_schema = self.llm_handler._get_json_schema_by_type("deduplication_response", "mistral")
        openai_schema = self.llm_handler._get_json_schema_by_type("deduplication_response", "openai")
        
        # Mistral: direct array
        assert mistral_schema["schema"]["type"] == "array"
        assert "items" in mistral_schema["schema"]
        
        # OpenAI: wrapped in events object
        assert openai_schema["schema"]["type"] == "object"
        assert "events" in openai_schema["schema"]["properties"]
        assert openai_schema["schema"]["properties"]["events"]["type"] == "array"

    def test_parse_mistral_event_extraction_response(self):
        """Test parsing Mistral event extraction response (direct array)"""
        # Simulate Mistral response (direct array)
        mistral_response = '''[
            {
                "source": "test_source",
                "dance_style": "swing",
                "url": "http://example.com",
                "event_type": "social",
                "event_name": "Test Dance",
                "day_of_week": "Friday",
                "start_date": "2025-01-01",
                "end_date": "2025-01-01",
                "start_time": "20:00",
                "end_time": "23:00",
                "price": "$10",
                "location": "Dance Studio",
                "description": "Fun dance event"
            }
        ]'''
        
        result = self.llm_handler.extract_and_parse_json(mistral_response, "test_url", "event_extraction")
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["event_name"] == "Test Dance"
        assert result[0]["dance_style"] == "swing"

    def test_parse_openai_event_extraction_response(self):
        """Test parsing OpenAI event extraction response (wrapped format)"""
        # Simulate OpenAI response (wrapped in events object)
        openai_response = '''{
            "events": [
                {
                    "source": "test_source",
                    "dance_style": "salsa",
                    "url": "http://example.com",
                    "event_type": "class",
                    "event_name": "Salsa Lessons",
                    "day_of_week": "Tuesday",
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-01",
                    "start_time": "19:00",
                    "end_time": "20:30",
                    "price": "Free",
                    "location": "Community Center",
                    "description": "Beginner salsa class"
                }
            ]
        }'''
        
        result = self.llm_handler.extract_and_parse_json(openai_response, "test_url", "event_extraction")
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["event_name"] == "Salsa Lessons"
        assert result[0]["dance_style"] == "salsa"

    def test_parse_address_extraction_response(self):
        """Test parsing address extraction response (single object)"""
        address_response = '''{
            "address_id": 123,
            "full_address": "123 Main St, Vancouver, BC, Canada",
            "building_name": "Dance Studio",
            "street_number": "123",
            "street_name": "Main St",
            "street_type": "Street",
            "direction": null,
            "city": "Vancouver",
            "met_area": null,
            "province_or_state": "BC",
            "postal_code": "V6A 1A1",
            "country_id": "CA",
            "time_stamp": "2025-01-01T10:00:00"
        }'''
        
        result = self.llm_handler.extract_and_parse_json(address_response, "test_url", "address_extraction")
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["address_id"] == 123
        assert result[0]["city"] == "Vancouver"

    def test_parse_openai_deduplication_response(self):
        """Test parsing OpenAI deduplication response (wrapped in events)"""
        dedup_response = '''{
            "events": [
                {
                    "group_id": 1,
                    "event_id": 101,
                    "Label": 0
                },
                {
                    "group_id": 1,
                    "event_id": 102,
                    "Label": 1
                }
            ]
        }'''
        
        result = self.llm_handler.extract_and_parse_json(dedup_response, "test_url", "deduplication_response")
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["event_id"] == 101
        assert result[1]["Label"] == 1

    def test_parse_mistral_deduplication_response(self):
        """Test parsing Mistral deduplication response (direct array)"""
        dedup_response = '''[
            {
                "group_id": 2,
                "event_id": 201,
                "Label": 0
            },
            {
                "group_id": 2,
                "event_id": 202,
                "Label": 1
            }
        ]'''
        
        result = self.llm_handler.extract_and_parse_json(dedup_response, "test_url", "deduplication_response")
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["group_id"] == 2
        assert result[1]["event_id"] == 202

    def test_parse_openai_address_deduplication_response(self):
        """Test parsing OpenAI address deduplication response (wrapped in addresses)"""
        addr_dedup_response = '''{
            "addresses": [
                {
                    "address_id": 301,
                    "Label": 0
                },
                {
                    "address_id": 302,
                    "Label": 1
                }
            ]
        }'''
        
        result = self.llm_handler.extract_and_parse_json(addr_dedup_response, "test_url", "address_deduplication")
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["address_id"] == 301
        assert result[1]["Label"] == 1

    def test_parse_malformed_json_fallback(self):
        """Test that malformed JSON falls back to line-based parsing"""
        malformed_response = '''This is not valid JSON but has some data:
        "event_name": "Test Event",
        "dance_style": "ballroom",
        "price": "$15"
        '''
        
        # Mock the line_based_parse method to return expected result
        with patch.object(self.llm_handler, 'line_based_parse') as mock_parse:
            mock_parse.return_value = [{"event_name": "Test Event", "dance_style": "ballroom"}]
            
            result = self.llm_handler.extract_and_parse_json(malformed_response, "test_url", "event_extraction")
            
            # Should call line_based_parse as fallback
            mock_parse.assert_called_once()
            assert result is not None
            assert result[0]["event_name"] == "Test Event"

    def test_parse_empty_response(self):
        """Test parsing empty or null responses"""
        # Test None
        result = self.llm_handler.extract_and_parse_json(None, "test_url", "event_extraction")
        assert result is None
        
        # Test "No events found"
        result = self.llm_handler.extract_and_parse_json("No events found", "test_url", "event_extraction")
        assert result is None
        
        # Test too short
        result = self.llm_handler.extract_and_parse_json("short", "test_url", "event_extraction")
        assert result is None

    def test_parse_mixed_content_detection(self):
        """Test content-based detection for unknown object types"""
        # Event-like object without schema_type
        event_obj = '{"event_name": "Mystery Dance", "start_time": "20:00"}'
        result = self.llm_handler.extract_and_parse_json(event_obj, "test_url", None)
        assert result is not None
        assert len(result) == 1
        assert result[0]["event_name"] == "Mystery Dance"
        
        # Address-like object without schema_type
        addr_obj = '{"address_id": 999, "full_address": "Mystery Location"}'
        result = self.llm_handler.extract_and_parse_json(addr_obj, "test_url", None)
        assert result is not None
        assert len(result) == 1
        assert result[0]["address_id"] == 999

    def test_schema_provider_parameter_handling(self):
        """Test that schema methods handle provider parameters correctly"""
        # Test default parameter
        mistral_default = self.llm_handler._get_json_schema_by_type("event_extraction")
        mistral_explicit = self.llm_handler._get_json_schema_by_type("event_extraction", "mistral")
        assert mistral_default == mistral_explicit
        
        # Test different providers give different results for event_extraction
        mistral_schema = self.llm_handler._get_json_schema_by_type("event_extraction", "mistral")
        openai_schema = self.llm_handler._get_json_schema_by_type("event_extraction", "openai")
        assert mistral_schema["schema"]["type"] == "array"
        assert openai_schema["schema"]["type"] == "object"

    def test_all_schema_types_exist(self):
        """Test that all expected schema types exist for both providers"""
        expected_schemas = [
            "event_extraction", 
            "address_extraction", 
            "deduplication_response", 
            "relevance_classification", 
            "address_deduplication"
        ]
        
        for schema_type in expected_schemas:
            mistral = self.llm_handler._get_json_schema_by_type(schema_type, "mistral")
            openai = self.llm_handler._get_json_schema_by_type(schema_type, "openai")
            
            assert mistral is not None, f"Mistral schema missing for {schema_type}"
            assert openai is not None, f"OpenAI schema missing for {schema_type}"
            assert "name" in mistral, f"Mistral schema {schema_type} missing name"
            assert "name" in openai, f"OpenAI schema {schema_type} missing name"
            assert "schema" in mistral, f"Mistral schema {schema_type} missing schema"
            assert "schema" in openai, f"OpenAI schema {schema_type} missing schema"


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def setup_method(self):
        """Setup test configuration"""
        self.config = {
            'llm': {
                'provider': 'mistral',
                'spend_money': True,
                'mistral_model': 'mistral-large-latest',
                'openai_model': 'gpt-4o-mini'
            },
            'crawling': {
                'prompt_max_length': 50000
            },
            'input': {
                'data_keywords': 'test_keywords.csv',
                'fb_urls': 'test_fb_urls.csv',
                'gs_urls': 'test_gs_urls.csv',
                'emails': 'test_emails.csv',
                'edge_cases': 'test_edge_cases.csv',
                'urls': 'test_urls'
            },
            'output': {
                'events': 'test_events.csv'
            },
            'prompts': {
                'default': 'prompts/default.txt'
            }
        }
        
        with patch('llm.Mistral'), patch('llm.OpenAI'), \
             patch('llm.DatabaseHandler'), \
             patch('builtins.open', mock_open()), \
             patch('llm.yaml.safe_load', return_value=self.config), \
             patch('llm.os.getenv', return_value='test_key'), \
             patch('llm.pd.read_csv', return_value=Mock(columns=['keyword'], to_list=Mock(return_value=['dance', 'swing']))):
            self.llm_handler = LLMHandler()

    def test_provider_fallback_parsing(self):
        """Test that parsing works regardless of which provider actually responds"""
        # Simulate scenario where Mistral fails, OpenAI succeeds with wrapped format
        openai_response = '{"events": [{"event_name": "Fallback Event", "dance_style": "tango"}]}'
        
        # Parser should handle OpenAI format even if we don't know which provider was used
        result = self.llm_handler.extract_and_parse_json(openai_response, "test_url", "event_extraction")
        
        assert result is not None
        assert len(result) == 1
        assert result[0]["event_name"] == "Fallback Event"

    def test_real_world_messy_response(self):
        """Test parsing real-world messy responses with cleanup"""
        messy_response = '''```json
        [
            {
                "source": "eventbrite", // This is a comment
                "dance_style": "swing",
                "event_name": "Friday Night Swing",
                "price": "$10",
                ...
                "location": "Main Hall"
            }
        ]
        ```'''
        
        with patch.object(self.llm_handler, 'line_based_parse') as mock_parse:
            mock_parse.return_value = [{"event_name": "Friday Night Swing", "dance_style": "swing"}]
            
            result = self.llm_handler.extract_and_parse_json(messy_response, "test_url", "event_extraction")
            
            # Should fall back to line-based parsing due to invalid JSON
            assert result is not None
            assert result[0]["event_name"] == "Friday Night Swing"


if __name__ == "__main__":
    # Run tests
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", __file__, "-v", "--tb=short"
    ], cwd=os.path.dirname(__file__))
    exit(result.returncode)