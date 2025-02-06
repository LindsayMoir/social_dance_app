import re

text = 'pgcal_inlineScript({"gcal":"17de2ca43f525c87058ffafe1f0206da82a19d00a85a6ae979ad6bb618dcd8ae@group.calendar.google.com","locale":"en","list_type":"listCustom","custom_list_button":"list","custom_days":"28","views":"dayGridMonth, listCustom","initial_view":"dayGridMonth","enforce_listview_on_mobile":"true","show_today_button":"true","show_title":"false","id_hash":"5ab0c7f65d","use_tooltip":"true","no_link":"true","fc_args":"{}"});'

# Extract other Google Calendar links from the response.text using regex
calendar_pattern = re.compile(r'"gcal"\s*:\s*"([a-zA-Z0-9_.+-]+@group\.calendar\.google\.com)"') 
cleaned_calendar_emails = calendar_pattern.findall(text)

# Debug log to verify extracted calendar emails
print(f"Extracted Google Calendar emails: {cleaned_calendar_emails}")

async def extract_shared_text(self, extracted_text, link):
        """
        Extracts text shared with the public from the given extracted text.
        This function uses a regular expression to find and extract the text that appears 
        between 'Shared with Public' and 'See more' in the provided extracted text.
        Args:
            extracted_text (str): The text from which to extract the shared text.
            link (str): The link associated with the extracted text, used for logging purposes.
        Returns:
            str or None: The extracted text if found, otherwise None.
        Logs:
            A warning if 'Shared with Public' or 'See more' is not found in the extracted text.
        """
        # Regex to find text after 'Shared with Public' up to 'See more'
        pattern = re.compile(r"Shared with Public\s*(.*?)\s*See more", re.IGNORECASE | re.DOTALL)

        match = pattern.search(extracted_text)
        
        if match:
            return match.group(1).strip()  # Extracted text between markers
        else:
            logging.warning(f"def extract_shared_text(): 'Shared with Public' or 'See more' not found in {link}.")
            return None