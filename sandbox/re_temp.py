import re

# Read this text file (extracted_text.txt) into extracted_text
with open('data/email_text.txt', 'r') as file:
    extracted_text = file.read()

match = re.search(r"classes at the  Legion.*?Victoria West Coast Swing Collective Society", extracted_text, re.DOTALL)

if match:
    extracted_text = match.group(0)  # Extract the first matchmatch = re.search(r"classes at the  Legion.*?Victoria West Coast Swing Collective Society", extracted_text, re.DOTALL)
    print(match.group(0))

else:
    print("fail")
