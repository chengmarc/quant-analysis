# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:55:08 2024

@author: Admin
"""
import os, sys, time, getpass, subprocess
script_path = os.path.dirname(os.path.realpath(__file__))

try:
    
    from selenium.webdriver import Firefox
    from selenium.webdriver import FirefoxProfile
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.firefox.service import Service as FirefoxService
    print("DRIVER: Webdriver initializer imported.")

except ImportError as e:
    print(f"DRIVER: The module '{e.name}' is not found, please install it using either pip or conda.")
    getpass.getpass("DRIVER: Press Enter to quit in a few seconds...")
    sys.exit()
    

# %%


def initialize_firefox(bit: int) -> Firefox:
    # Current webdriver version:
    # mozilla/geckodriver 0.33.0

    options = FirefoxOptions()
    #options.add_argument("-headless")
    options.add_argument("--width=800")
    options.add_argument("--height=600")
    
    service = FirefoxService()
    service.creation_flags = subprocess.CREATE_NO_WINDOW
    
    profile = FirefoxProfile()
    profile.set_preference("dom.popup_maximum", 40000) # Set the maximum number of pop-ups to 0
    profile.set_preference("dom.popup_allowed_events", "click") # Allow pop-ups on click
    profile.set_preference("dom.popup_allowed_events", "change") # Allow pop-ups on change
    profile.update_preferences()

    if bit == 32:
        binary = os.path.join(script_path, "geckodriver32.exe")
        driver = Firefox(executable_path=binary, firefox_profile=profile, service=service, options=options)
    if bit == 64:
        binary = os.path.join(script_path, "geckodriver64.exe")
        driver = Firefox(executable_path=binary, firefox_profile=profile, service=service, options=options)
    return driver


# %%


for i in range(350):

    with open(os.path.join(script_path, "links", f"links{i}.html"), "w") as file:
        file.write("<html>\n")
        file.write("<body>\n")
        for j in range(i*100+1, (i+1)*100+1):
            link = f"https://www.coingecko.com/price_charts/export/{j}/usd.csv"
            file.write(f"<a href='{link}'>{link}</a><br>\n")
        file.write("</body>\n")
        file.write("</html>\n")

    driver = initialize_firefox(64)

    driver.get(os.path.join(script_path, "links", f"links{i}.html"))
    links = driver.find_elements("css selector", "a")
    for url in links:
        driver.execute_script("window.open(arguments[0], '_blank');", url.get_attribute("href"))
        print("Processed", url.get_attribute("href"))
        time.sleep(1)
    time.sleep(5)

    driver.quit()
    
    os.remove(os.path.join(script_path, "links", f"links{i}.html"))

