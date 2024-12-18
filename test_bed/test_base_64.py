import base64

calendar_id = '75b7agh1gm6fm5jom929ngrl44'

# Add padding if necessary
missing_padding = len(calendar_id) % 4
if missing_padding != 0:
    calendar_id += '=' * (4 - missing_padding)

try:
    # Decode the string safely
    decoded_bytes = base64.urlsafe_b64decode(calendar_id)
    # Try decoding as UTF-8
    calendar_id_utf8 = decoded_bytes.decode('utf-8')
    print("Decoded as UTF-8:", calendar_id_utf8)
except UnicodeDecodeError:
    print("Decoding as UTF-8 failed, attempting different decoding.")
    # If UTF-8 decoding fails, print the raw bytes
    print("Raw decoded bytes:", decoded_bytes)

        # Regular expression to find calendar IDs
        calendar_id_pattern = r'src=([^&]+%40group.calendar.google.com)'

        # Find all calendar IDs
        calendar_ids = re.findall(calendar_id_pattern, calendar_url)

        # Decode the calendar IDs
        decoded_calendar_ids = [id.replace('%40', '@') for id in calendar_ids]

        # Print the extracted calendar IDs
        for calendar_id in decoded_calendar_ids:
            print(calendar_id)