import re

text = 'pgcal_inlineScript({"gcal":"17de2ca43f525c87058ffafe1f0206da82a19d00a85a6ae979ad6bb618dcd8ae@group.calendar.google.com","locale":"en","list_type":"listCustom","custom_list_button":"list","custom_days":"28","views":"dayGridMonth, listCustom","initial_view":"dayGridMonth","enforce_listview_on_mobile":"true","show_today_button":"true","show_title":"false","id_hash":"5ab0c7f65d","use_tooltip":"true","no_link":"true","fc_args":"{}"});'

# Extract other Google Calendar links from the response.text using regex
calendar_pattern = re.compile(r'"gcal"\s*:\s*"([a-zA-Z0-9_.+-]+@group\.calendar\.google\.com)"') 
cleaned_calendar_emails = calendar_pattern.findall(text)

# Debug log to verify extracted calendar emails
print(f"Extracted Google Calendar emails: {cleaned_calendar_emails}")