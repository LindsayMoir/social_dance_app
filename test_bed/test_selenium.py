from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import chromedriver_autoinstaller

# Automatically install the correct version of ChromeDriver
chromedriver_autoinstaller.install()

# Set up Chrome options
options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')  # Useful for Linux systems
options.add_argument('--disable-dev-shm-usage')  # Prevents crashes in low-resource environments
options.add_argument('--headless')  # Run in headless mode if desired (omit for visible mode)

# Initialize the WebDriver
driver = webdriver.Chrome(options=options)

# Test with Google
driver.get("https://www.google.com")
print(driver.title)
driver.quit()
