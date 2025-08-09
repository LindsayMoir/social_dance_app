"""
Test script to verify that prompt_type parameter fixes work correctly throughout the codebase.
This will test the generate_prompt and process_llm_response methods with various prompt types.
"""

import sys
import os
import yaml
import logging

# Add the src directory to the Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm import LLMHandler

def test_generate_prompt():
    """Test the generate_prompt method with various prompt types."""
    print("\n=== Testing generate_prompt method ===")
    
    # Initialize LLM handler
    llm_handler = LLMHandler()
    
    # Test cases: (prompt_type, expected_behavior)
    test_cases = [
        ('fb', 'should work - fb prompt exists'),
        ('default', 'should work - default prompt exists'),
        ('https://gotothecoda.com/calendar', 'should work - URL-specific prompt exists'),
        ('https://www.bardandbanker.com/live-music', 'should work - URL-specific prompt exists'),
        ('https://www.debrhymerband.com/shows', 'should work - URL-specific prompt exists'),
        ('https://vbds.org/other-dancing-opportunities/', 'should work - URL-specific prompt exists'),
        ('images', 'should work - images prompt exists'),
        ('irrelevant_rows', 'should work - irrelevant_rows prompt exists'),
        ('nonexistent_prompt', 'should fallback to default with warning'),
        ('single_event', 'should work - single_event prompt exists'),
    ]
    
    for prompt_type, expected in test_cases:
        try:
            print(f"\nTesting prompt_type: '{prompt_type}' ({expected})")
            prompt_text, schema_type = llm_handler.generate_prompt(
                url="test_url", 
                extracted_text="test extracted text", 
                prompt_type=prompt_type
            )
            
            print(f"  ✓ Success: Got prompt_text length={len(prompt_text)}, schema_type={schema_type}")
            
            # Verify it's actually a string with content
            if not isinstance(prompt_text, str) or len(prompt_text) < 10:
                print(f"  ⚠ Warning: Prompt text seems too short or invalid")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")

def test_process_llm_response():
    """Test the process_llm_response method (but skip actual LLM calls)."""
    print("\n=== Testing process_llm_response method (parameter validation) ===")
    
    # Initialize LLM handler
    llm_handler = LLMHandler()
    
    # Temporarily disable spending money to avoid actual LLM calls
    original_spend_money = llm_handler.config['llm']['spend_money']
    llm_handler.config['llm']['spend_money'] = False
    
    test_cases = [
        'fb',
        'default', 
        'https://gotothecoda.com/calendar',
        'nonexistent_prompt'
    ]
    
    for prompt_type in test_cases:
        try:
            print(f"\nTesting process_llm_response with prompt_type: '{prompt_type}'")
            
            # This should not error on the parameter passing, even if it returns None due to spend_money=False
            result = llm_handler.process_llm_response(
                url="test_url",
                parent_url="test_parent_url", 
                extracted_text="test extracted text",
                source="test_source",
                keywords_list=['dance'],
                prompt_type=prompt_type
            )
            
            print(f"  ✓ Parameter passing works (result: {result})")
            
        except Exception as e:
            print(f"  ✗ Error in parameter handling: {e}")
    
    # Restore original setting
    llm_handler.config['llm']['spend_money'] = original_spend_money

def test_config_prompt_mappings():
    """Test that all prompt mappings in config are valid."""
    print("\n=== Testing config prompt mappings ===")
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    prompts = config.get('prompts', {})
    print(f"Found {len(prompts)} prompt mappings in config")
    
    for prompt_key, prompt_config in prompts.items():
        try:
            if isinstance(prompt_config, dict):
                # New format
                file_path = prompt_config.get('file')
                schema_type = prompt_config.get('schema')
                print(f"  {prompt_key}: file='{file_path}', schema='{schema_type}'")
                
                # Check if file exists
                full_path = os.path.join(os.path.dirname(__file__), '..', file_path)
                if os.path.exists(full_path):
                    print(f"    ✓ File exists")
                else:
                    print(f"    ✗ File missing: {full_path}")
            else:
                # Old format  
                print(f"  {prompt_key}: legacy format '{prompt_config}'")
                full_path = os.path.join(os.path.dirname(__file__), '..', prompt_config)
                if os.path.exists(full_path):
                    print(f"    ✓ File exists")
                else:
                    print(f"    ✗ File missing: {full_path}")
                    
        except Exception as e:
            print(f"  ✗ Error processing {prompt_key}: {e}")

def test_method_signatures():
    """Test that method signatures are consistent."""
    print("\n=== Testing method signatures ===")
    
    try:
        llm_handler = LLMHandler()
        
        # Test generate_prompt signature
        import inspect
        sig = inspect.signature(llm_handler.generate_prompt)
        params = list(sig.parameters.keys())
        print(f"generate_prompt parameters: {params}")
        
        if 'prompt_type' in params:
            print("  ✓ generate_prompt has prompt_type parameter")
        else:
            print("  ✗ generate_prompt missing prompt_type parameter")
            
        # Test process_llm_response signature  
        sig = inspect.signature(llm_handler.process_llm_response)
        params = list(sig.parameters.keys())
        print(f"process_llm_response parameters: {params}")
        
        if 'prompt_type' in params:
            print("  ✓ process_llm_response has prompt_type parameter")
        else:
            print("  ✗ process_llm_response missing prompt_type parameter")
            
    except Exception as e:
        print(f"  ✗ Error checking signatures: {e}")

if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )
    
    print("Testing prompt_type parameter fixes throughout codebase...")
    
    test_config_prompt_mappings()
    test_method_signatures() 
    test_generate_prompt()
    test_process_llm_response()
    
    print("\n=== Test Summary ===")
    print("✓ Tests completed - check output above for any issues")
    print("✓ If no errors shown, the prompt_type fixes are working correctly")
    print("✗ If errors found, they need to be addressed before running the pipeline")