# Prompt Mapping System Documentation

## Overview

The social dance app uses a flexible prompt mapping system that allows different websites and contexts to use specialized prompts for LLM processing. This system supports both simple key-based prompts and URL-based prompts for site-specific customization.

## How It Works

The `prompt_type` parameter is used throughout the codebase to specify which prompt to use. The `generate_prompt()` method in `src/llm.py` maps these prompt types to actual prompt files using the configuration in `config/config.yaml`.

## Prompt Type Formats

### 1. Simple Key Prompts

These are straightforward string keys that map to prompt files:

```python
prompt_type = 'fb'           # → prompts/fb_prompt.txt
prompt_type = 'default'      # → prompts/default.txt  
prompt_type = 'images'       # → prompts/images_prompt.txt
prompt_type = 'dedup'        # → prompts/dedup_prompt.txt
prompt_type = 'irrelevant_rows' # → prompts/irrelevant_rows_prompt.txt
```

### 2. URL-Based Prompts (Site-Specific)

Full URLs can be used as prompt types to enable site-specific customization:

```python
prompt_type = 'https://gotothecoda.com/calendar'        # → prompts/the_coda_prompt.txt
prompt_type = 'https://www.bardandbanker.com/live-music' # → prompts/bard_and_banker_prompt.txt
prompt_type = 'https://www.debrhymerband.com/shows'     # → prompts/deb_rhymer_prompt.txt
prompt_type = 'https://vbds.org/other-dancing-opportunities/' # → prompts/default.txt
```

## Configuration

The mapping is defined in `config/config.yaml` under the `prompts` section:

```yaml
prompts:
  fb:
    file: prompts/fb_prompt.txt
    schema: event_extraction
  default:
    file: prompts/default.txt
    schema: event_extraction
  https://gotothecoda.com/calendar:
    file: prompts/the_coda_prompt.txt
    schema: event_extraction
  https://www.bardandbanker.com/live-music:
    file: prompts/bard_and_banker_prompt.txt
    schema: event_extraction
```

## Usage Patterns in Code

### Pattern 1: Static Prompt Type
```python
# For Facebook pages
prompt_type = 'fb'
llm_handler.process_llm_response(url, parent_url, extracted_text, source, keywords, prompt_type)
```

### Pattern 2: URL-Based Prompt Type  
```python
# For site-specific processing (rd_ext.py pattern)
prompt_type = url  # URL itself is used as prompt_type
llm_handler.process_llm_response(event_url, parent_url, text, source, keywords, prompt_type)
```

### Pattern 3: Generate Prompt
```python
# Generate the actual prompt text
prompt_text, schema_type = llm_handler.generate_prompt(url, extracted_text, prompt_type)
```

## Fallback Behavior

If a `prompt_type` is not found in the configuration:
1. The system logs a warning: `"Prompt type 'X' not found, using default"`
2. It falls back to using the 'default' prompt
3. Processing continues normally

## Adding New Prompt Types

To add a new prompt type:

1. **Create the prompt file**: Add your prompt text to a file in the `prompts/` directory
2. **Update config.yaml**: Add the mapping under the `prompts` section
3. **Use in code**: Pass the prompt type to `generate_prompt()` or `process_llm_response()`

Example:
```yaml
prompts:
  my_new_site:
    file: prompts/my_new_site_prompt.txt
    schema: event_extraction
```

```python
prompt_type = 'my_new_site'
result = llm_handler.process_llm_response(url, parent_url, text, source, keywords, prompt_type)
```

## Files That Use prompt_type Parameter

- `src/llm.py` - Core prompt mapping logic
- `src/ebs.py` - Eventbrite processing  
- `src/fb.py` - Facebook processing
- `src/scraper.py` - General web scraping
- `src/rd_ext.py` - URL-specific processing
- `src/images.py` - Image processing
- `src/emails.py` - Email processing
- `src/clean_up.py` - Data cleanup operations

## Schema Types

Each prompt can specify a JSON schema type for structured output:

- `event_extraction` - For extracting event data
- `address_extraction` - For extracting address information
- `deduplication_response` - For deduplication operations  
- `relevance_classification` - For relevance scoring
- `address_deduplication` - For address deduplication
- `null` - No structured output expected

## Best Practices

1. **Use descriptive prompt types**: Choose clear, meaningful names
2. **Document new prompts**: Add comments explaining the purpose
3. **Test fallback behavior**: Ensure your code works when prompt_type isn't found
4. **Use URL-based prompts sparingly**: Only when site-specific customization is truly needed
5. **Keep prompts organized**: Group related prompts in the config file