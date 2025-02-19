import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

print(os.getenv("RENDER_INTERNAL_DB_URL"))  # Ensure it prints the correct URL

print(os.getenv('ADDRESS_DB_CONNECTION_STRING'))