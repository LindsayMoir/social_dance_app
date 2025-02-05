import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import re

async def scrape_keyword_context(url, output_file='keyword_context.csv'):
    keywords = ['google', 'calendar', 'pretty']
    # Compile a regex pattern to match the keywords with word boundaries, case-insensitive
    pattern = re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
    contexts = []

    async with async_playwright() as p:
        # Launch the browser in headful mode
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        # Navigate to the target URL
        await page.goto(url, timeout=60000)
        
        # Retrieve the page content without altering its case
        content = await page.content()
        
        # Close the browser
        await browser.close()

    # Iterate over all matches of the keywords in the content
    for match in pattern.finditer(content):
        start, end = match.span()
        keyword = match.group(0)
        
        # Extract 100 characters before the keyword, handling edge cases
        context_before = content[max(start-100, 0):start]
        
        # Extract 100 characters after the keyword, handling edge cases
        context_after = content[end:end+100]
        
        # Combine the contexts with the keyword
        full_context = context_before + keyword + context_after
        
        # Append the extracted data to the contexts list
        contexts.append({
            'Keyword': keyword,
            'Context_Before': context_before,
            'Context_After': context_after,
            'Full_Context': full_context
        })

    # Create a pandas DataFrame from the contexts list
    df = pd.DataFrame(contexts)
    
    # Write the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    
    print(f"Keyword contexts saved to {output_file}")

if __name__ == "__main__":
    target_url = 'https://vlda.ca/resources/'  # Updated URL
    asyncio.run(scrape_keyword_context(target_url))
