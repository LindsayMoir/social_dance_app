import re
from bs4 import BeautifulSoup

def extract_calendar_related_info(html_source):
    """
    Extract potential calendar-related information from the HTML source.
    This includes iframe sources, links, scripts, and mentions of calendars or events.
    """
    soup = BeautifulSoup(html_source, "html.parser")
    calendar_urls = set()

    # Look for iframes
    for iframe in soup.find_all("iframe", src=True):
        src = iframe["src"]
        calendar_urls.add(src)

    # Look for links with hrefs
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if any(keyword in href.lower() for keyword in ["calendar", "event", "google", "outlook", "ics", "ical"]):
            calendar_urls.add(href)

    # Look for inline scripts or JSON with URLs
    for script in soup.find_all("script"):
        script_content = script.string
        if script_content:
            # Extract URLs from the script content
            urls_in_script = re.findall(r"https?://[^\s\"']+", script_content)
            for url in urls_in_script:
                if any(keyword in url.lower() for keyword in ["calendar", "event", "google", "outlook", "ics", "ical"]):
                    calendar_urls.add(url)

    # Look for meta tags that might reference calendars
    for meta in soup.find_all("meta", content=True):
        content = meta["content"]
        if any(keyword in content.lower() for keyword in ["calendar", "event", "google", "outlook", "ics", "ical"]):
            calendar_urls.add(content)

    # Look for text that might hint at calendar systems
    calendar_related_text = []
    for text in soup.stripped_strings:
        if any(keyword in text.lower() for keyword in ["calendar", "google", "outlook", "event", "schedule", "ics", "ical"]):
            calendar_related_text.append(text)

    return calendar_urls, calendar_related_text

# Example Usage
if __name__ == "__main__":
    # Load the HTML source (replace 'bard_and_banker_source.html' with your actual file or string)
    with open("data/other/bb_source.txt", "r", encoding="utf-8") as file:
        html_source = file.read()

    # Extract calendar-related URLs and text
    calendar_urls, calendar_related_text = extract_calendar_related_info(html_source)

    # Display the extracted URLs
    print("=== Found Calendar-Related URLs ===")
    if calendar_urls:
        for url in calendar_urls:
            print(url)
    else:
        print("No calendar URLs found.")

    # Display any related text that might hint at calendars
    print("\n=== Found Calendar-Related Text ===")
    if calendar_related_text:
        for text in calendar_related_text:
            print(text)
    else:
        print("No calendar-related text found.")
