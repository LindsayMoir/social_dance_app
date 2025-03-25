import requests
from bs4 import BeautifulSoup

def decode_secret_message(url):
    """
    Retrieves a Google Doc from the given URL (which is in HTML format),
    parses the table containing the grid data, and prints the grid.
    
    The document is assumed to have a header row (with column names like
    "x-coordinate", "Character", "y-coordinate"), followed by data rows.
    
    The grid is built such that (0, 0) is at the bottom left.
    """
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve data from URL.")
        return

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all table rows
    rows = soup.find_all('tr')
    if not rows:
        print("No rows found in the document.")
        return

    # Determine if the first row is a header by checking for non-numeric text
    header_cells = rows[0].find_all(['td', 'th'])
    header_text = [cell.get_text(strip=True).lower() for cell in header_cells]
    if header_text and any("coordinate" in text for text in header_text):
        data_rows = rows[1:]
    else:
        data_rows = rows

    points = []
    max_x = 0
    max_y = 0

    # Process each row: expect three cells (x-coordinate, Character, y-coordinate)
    for row in data_rows:
        cells = row.find_all('td')
        if len(cells) < 3:
            continue  # Skip rows that don't have enough data
        try:
            x = int(cells[0].get_text(strip=True))
            char = cells[1].get_text(strip=True)
            y = int(cells[2].get_text(strip=True))
        except ValueError:
            continue  # Skip rows with invalid data
        points.append((x, y, char))
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    # Initialize the grid with spaces.
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Place the characters in the grid based on their (x, y) coordinates.
    for x, y, char in points:
        grid[y][x] = char

    # Print the grid from top (max_y) to bottom (0) so that (0, 0) appears at the bottom-left.
    for row in reversed(grid):
        print("".join(row))

if __name__ == "__main__":
    url = "https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub"
    decode_secret_message(url)
