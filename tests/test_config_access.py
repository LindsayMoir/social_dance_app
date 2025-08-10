#!/usr/bin/env python3
"""
Test to verify configuration access patterns still work correctly.
"""

import sys
import os
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm import LLMHandler
import yaml

def test_prompt_generation():
    """Test that prompt generation still works correctly."""
    print("\n=== Testing Prompt Generation ===")
    
    llm_handler = LLMHandler()
    
    # Test basic prompt types
    test_cases = [
        ("default", "https://example.com", "test event text"),
        ("fb", "https://facebook.com/event", "facebook event text"),
        ("address_internet_fix", "address_fix", "123 Main St, Vancouver")
    ]
    
    success_count = 0
    for prompt_type, url, text in test_cases:
        try:
            prompt_text, schema_type = llm_handler.generate_prompt(url, text, prompt_type)
            
            if prompt_text and isinstance(prompt_text, str) and len(prompt_text) > 0:
                print(f"✓ {prompt_type}: Generated prompt ({len(prompt_text)} chars), schema: {schema_type}")
                success_count += 1
            else:
                print(f"✗ {prompt_type}: Failed to generate prompt")
                
        except Exception as e:
            print(f"✗ {prompt_type}: Exception - {e}")
    
    return success_count == len(test_cases)

def test_config_loading():
    """Test that configuration loading works."""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        llm_handler = LLMHandler()
        
        # Check that config was loaded
        if hasattr(llm_handler, 'config') and llm_handler.config:
            print("✓ Configuration loaded successfully")
            
            # Check prompts section
            if 'prompts' in llm_handler.config:
                prompt_count = len(llm_handler.config['prompts'])
                print(f"✓ Found {prompt_count} prompt configurations")
                
                # Check a few key prompts exist
                key_prompts = ['default', 'fb']
                found_prompts = [p for p in key_prompts if p in llm_handler.config['prompts']]
                
                if len(found_prompts) == len(key_prompts):
                    print(f"✓ Key prompts found: {found_prompts}")
                    return True
                else:
                    print(f"✗ Missing key prompts: {set(key_prompts) - set(found_prompts)}")
                    return False
            else:
                print("✗ No prompts section in config")
                return False
        else:
            print("✗ Configuration not loaded")
            return False
            
    except Exception as e:
        print(f"✗ Exception loading configuration: {e}")
        return False

def test_config_access():
    """Run configuration access tests."""
    print("⚙️  TESTING CONFIGURATION ACCESS")
    print("=" * 35)
    
    # Suppress detailed logging
    logging.getLogger().setLevel(logging.ERROR)
    
    tests = [
        ("Configuration loading", test_config_loading()),
        ("Prompt generation", test_prompt_generation())
    ]
    
    print(f"\n{'=' * 35}")
    print("📊 CONFIGURATION ACCESS RESULTS") 
    print(f"{'=' * 35}")
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 RESULT: {passed}/{len(tests)} configuration tests passed")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = test_config_access()
    sys.exit(0 if success else 1)