"""
Test actual method calls to ensure parameter passing works correctly.
This simulates the actual calling patterns from the codebase.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm import LLMHandler

def test_llm_handler_methods():
    """Test the key methods that were fixed."""
    print("\n=== Testing LLM Handler Method Calls ===")
    
    llm_handler = LLMHandler()
    
    # Disable spending money to avoid actual API calls
    llm_handler.config['llm']['spend_money'] = False
    
    # Test cases that simulate actual usage patterns from the codebase
    test_cases = [
        {
            'name': 'Facebook URL processing (from fb.py pattern)',
            'prompt_type': 'fb',
            'url': 'https://facebook.com/events/123',
            'text': 'Sample Facebook event text with dance keywords'
        },
        {
            'name': 'URL-specific prompt (from rd_ext.py pattern)',
            'prompt_type': 'https://gotothecoda.com/calendar', 
            'url': 'https://gotothecoda.com/calendar/event/123',
            'text': 'Sample coda event text'
        },
        {
            'name': 'Default prompt (from scraper.py pattern)',
            'prompt_type': 'default',
            'url': 'https://example.com/events',
            'text': 'Sample generic event text'
        },
        {
            'name': 'Image processing (from images.py pattern)',
            'prompt_type': 'images',
            'url': 'https://example.com/image.jpg',
            'text': 'Sample image text extraction'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        try:
            # Test generate_prompt (like in clean_up.py, ebs.py)
            print("Testing generate_prompt...")
            prompt_text, schema_type = llm_handler.generate_prompt(
                url=test_case['url'],
                extracted_text=test_case['text'],
                prompt_type=test_case['prompt_type']
            )
            print(f"  ✓ generate_prompt works: len={len(prompt_text)}, schema={schema_type}")
            
            # Test process_llm_response (like in fb.py, scraper.py)
            print("Testing process_llm_response...")
            result = llm_handler.process_llm_response(
                url=test_case['url'],
                parent_url='test_parent',
                extracted_text=test_case['text'],
                source='test_source',
                keywords_list=['dance'],
                prompt_type=test_case['prompt_type']
            )
            print(f"  ✓ process_llm_response works: result={result}")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

def test_driver_method():
    """Test the driver method which was one of the first places we fixed."""
    print("\n=== Testing Driver Method ===")
    
    llm_handler = LLMHandler()
    llm_handler.config['llm']['spend_money'] = False
    
    try:
        # This simulates the actual call pattern from driver()
        result = llm_handler.driver(
            url='https://facebook.com/events/123',
            search_term='dance events',
            extracted_text='This is a salsa dance event happening tonight',
            source='Facebook',
            keywords_list=['salsa', 'dance']
        )
        print(f"✓ driver method works: result={result}")
        
    except Exception as e:
        print(f"✗ ERROR in driver: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing actual method call patterns...")
    
    test_llm_handler_methods()
    test_driver_method()
    
    print("\n=== Final Summary ===")
    print("If no errors above, the prompt_type fixes are working correctly!")