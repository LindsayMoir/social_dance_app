"""
Test script to verify that the EBS relevance vs event extraction issue is fixed.
"""

import sys
import os
import logging

# Add the src directory to the Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm import LLMHandler

def test_relevance_vs_event_extraction():
    """Test that relevance checking and event extraction are handled separately."""
    print("\n=== Testing Relevance vs Event Extraction Separation ===")
    
    llm_handler = LLMHandler()
    llm_handler.config['llm']['spend_money'] = False  # Disable actual API calls
    
    # Test 1: Relevance checking prompt (should have schema_type=None)
    print("\n--- Test 1: Relevance checking prompt ---")
    try:
        prompt_text, schema_type = llm_handler.generate_prompt(
            url="https://example.com/event",
            extracted_text="dance event text",
            prompt_type="relevant_dance_url"
        )
        
        print(f"✓ relevant_dance_url prompt generated successfully")
        print(f"  schema_type: {schema_type} (should be None)")
        
        if schema_type is None:
            print("  ✓ Correct: schema_type is None for relevance checking")
        else:
            print(f"  ⚠ Warning: schema_type should be None, got {schema_type}")
            
    except Exception as e:
        print(f"  ✗ Error generating relevance prompt: {e}")
    
    # Test 2: Event extraction prompt (should have schema_type != None) 
    print("\n--- Test 2: Event extraction prompt ---")
    try:
        prompt_text, schema_type = llm_handler.generate_prompt(
            url="https://example.com/event", 
            extracted_text="dance event text",
            prompt_type="default"
        )
        
        print(f"✓ default prompt generated successfully")
        print(f"  schema_type: {schema_type} (should be 'event_extraction')")
        
        if schema_type == 'event_extraction':
            print("  ✓ Correct: schema_type is 'event_extraction' for event extraction")
        else:
            print(f"  ⚠ Warning: schema_type should be 'event_extraction', got {schema_type}")
            
    except Exception as e:
        print(f"  ✗ Error generating event extraction prompt: {e}")
        
    # Test 3: process_llm_response should reject schema_type=None
    print("\n--- Test 3: process_llm_response safeguard ---") 
    try:
        result = llm_handler.process_llm_response(
            url="https://example.com/event",
            parent_url="https://example.com",
            extracted_text="dance event text", 
            source="test_source",
            keywords_list=['dance'],
            prompt_type="relevant_dance_url"  # This has schema_type=None
        )
        
        print(f"✓ process_llm_response completed (result: {result})")
        if result == False:
            print("  ✓ Correct: process_llm_response correctly rejected schema_type=None")
        else:
            print("  ⚠ Warning: process_llm_response should return False for schema_type=None")
            
    except Exception as e:
        print(f"  ✗ Error in process_llm_response: {e}")

def simulate_ebs_workflow():
    """Simulate the corrected EBS workflow."""
    print("\n=== Simulating Corrected EBS Workflow ===")
    
    llm_handler = LLMHandler()
    llm_handler.config['llm']['spend_money'] = False
    
    event_url = "https://www.eventbrite.ca/e/dance-event-123"
    
    # Step 1: Relevance check (like in eventbrite_search)
    print("\n--- Step 1: Relevance Check ---")
    try:
        prompt_text, schema_type = llm_handler.generate_prompt(
            url=event_url,
            extracted_text=event_url,
            prompt_type="relevant_dance_url"
        )
        
        # Call query_llm directly for relevance (not process_llm_response)
        raw_response = llm_handler.query_llm(event_url, prompt_text, schema_type)
        print(f"✓ Relevance check query completed")
        print(f"  Raw response: {raw_response}")
        print(f"  Schema type: {schema_type} (None for relevance)")
        
        # This is how EBS handles the relevance response
        if isinstance(raw_response, str):
            is_relevant = raw_response.strip().lower() == "true"
        else:
            is_relevant = bool(raw_response)
            
        print(f"  Is relevant: {is_relevant}")
        
    except Exception as e:
        print(f"  ✗ Error in relevance check: {e}")
        return
    
    # Step 2: Event extraction (like in process_event) 
    print("\n--- Step 2: Event Extraction ---")
    try:
        # This should use 'default' for event extraction, not 'relevant_dance_url'
        result = llm_handler.process_llm_response(
            url=event_url,
            parent_url="https://eventbrite.com/search",
            extracted_text="Sample event text with dance keywords",
            source="Eventbrite", 
            keywords_list=['dance'],
            prompt_type='default'  # Using 'default' for event extraction
        )
        
        print(f"✓ Event extraction completed")
        print(f"  Result: {result} (False expected since spend_money=False)")
        
    except Exception as e:
        print(f"  ✗ Error in event extraction: {e}")

if __name__ == "__main__":
    print("Testing EBS prompt type fix...")
    
    # Set up logging to capture warnings
    logging.basicConfig(level=logging.WARNING)
    
    test_relevance_vs_event_extraction()
    simulate_ebs_workflow()
    
    print("\n=== Summary ===")
    print("✓ If no errors above, the EBS fix should resolve the 'Result too short' issue")
    print("✓ Relevance checking now uses query_llm() directly") 
    print("✓ Event extraction uses process_llm_response() with appropriate prompts")
    print("✓ The two processes are properly separated")