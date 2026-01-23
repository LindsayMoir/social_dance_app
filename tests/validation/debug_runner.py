#!/usr/bin/env python3
"""Debug script to test validation runner"""

import sys
import os

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(repo_root, 'src'))
sys.path.insert(0, script_dir)

print("Script starting...")
print(f"Script dir: {script_dir}")
print(f"Repo root: {repo_root}")
print(f"sys.path: {sys.path[:3]}")

from dotenv import load_dotenv
print("✓ Imported dotenv")

load_dotenv('src/.env')
print("✓ Loaded .env")

from datetime import datetime
import logging
import yaml
print("✓ Imported standard libs")

from scraping_validator import ScrapingValidator
print("✓ Imported scraping_validator")

from chatbot_evaluator import (
    TestQuestionGenerator,
    ChatbotTestExecutor,
    ChatbotScorer,
    generate_chatbot_report
)
print("✓ Imported chatbot_evaluator")

from db import DatabaseHandler
print("✓ Imported db")

from llm import LLMHandler
print("✓ Imported llm")

from logging_config import setup_logging
print("✓ Imported logging_config")

# Setup logging
setup_logging('validation_tests')
print("✓ Setup logging")

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print("✓ Loaded config")

# Initialize handlers
db_handler = DatabaseHandler(config)
print("✓ Initialized db_handler")

llm_handler = LLMHandler('config/config.yaml')
print("✓ Initialized llm_handler")

print("\nAll initialization successful!")
print("Now testing a simple query...")

# Test date calculator
from date_calculator import calculate_date_range
result = calculate_date_range('next month', '2026-01-22')
print(f"✓ Date calculator works: {result}")

print("\nDebug test completed successfully!")

print("\n" + "="*70)
print("TESTING SINGLE QUESTION")
print("="*70)

# Create test executor
executor = ChatbotTestExecutor(
    config=config,
    db_handler=db_handler
)
print("✓ Created ChatbotTestExecutor")

# Test with a simple question
test_question = {
    'question': 'Show me salsa events next month',
    'category': 'test',
    'parameters': {},
    'expected_criteria': {'dance_style': 'salsa', 'timeframe': 'next month'}
}

print(f"\nTesting question: {test_question['question']}")

# First, test the SQL syntax check directly
test_sql = """SELECT
    event_name,
    event_type,
    dance_style,
    day_of_week,
    start_date,
    end_date,
    start_time,
    end_time,
    source,
    url,
    price,
    description,
    location
FROM
    events
WHERE
    dance_style ILIKE '%salsa%'
    AND start_date >= '2026-02-01'
    AND start_date <= '2026-02-28'
ORDER BY
    start_date, start_time
LIMIT 30"""

print("\nTesting _check_sql_syntax directly...")
syntax_check = executor._check_sql_syntax(test_sql)
print(f"Syntax check result: {syntax_check}")

print("\nNow running full test...")
result = executor.execute_test_question(test_question)
print(f"✓ Test executed")
print(f"\nFull SQL:\n{result.get('sql_query', '')}")
print(f"\nExecution success: {result.get('execution_success')}")
print(f"Result count: {result.get('result_count')}")
print(f"SQL syntax valid: {result.get('sql_syntax_valid')}")
if 'error_message' in result:
    print(f"Error message: {result['error_message']}")

print("\nSingle question test completed!")
