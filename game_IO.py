from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import pyautogui
import time
import numpy as np

game_region = (576, 213, 746, 748)
score_region = (647, 243, 75, 36)

class IO():
    def options(self):
        option = Options()
        option.add_argument("--disable-infobars")
        option.add_argument("start-maximized")
        option.add_argument("--disable-extensions")
        # option.add_argument("--headless")
        option.add_experimental_option("prefs", {
            "profile.default_content_setting_values.notifications": 2
        })
        return option

    def initialize(self):
        DRIVER_PATH = r'C:\Users\Levi\Programming\Python\Deep_Learning\Snake\chromedriver.exe'

        chrome = webdriver.Chrome(options=self.options(), executable_path=DRIVER_PATH)
        chrome.get('https://www.google.com/search?q=play+snake')

        self._driver = chrome

        play_button = chrome.find_element_by_xpath('//*[@id="rso"]/div[1]/div[1]/div/div[1]/div/div/div/div[1]/div[2]/div/div')
        play_button.click()
        time.sleep(1)
        # self.do_keystroke(Keys.SPACE)
        # time.sleep(1)
        # self.canvas = chrome.find_element_by_xpath('//*[@id="rso"]/div[1]/div[1]/div/div[1]/div/div/div/g-lightbox/div[2]/div[2]/span/div/div[2]/canvas')

    def do_action(self, action):
        key = None
        if action == 0:
            key = Keys.LEFT
        elif action == 1:
            key = Keys.UP
        elif action == 2:
            key = Keys.RIGHT
        elif action == 3:
            key = Keys.DOWN
        elif action == 4: # do nothing
            return
        else:
            raise ValueError('`action` should be between 0 and 3, was ' + action)

        ActionChains(self._driver) \
            .key_down(key) \
            .key_up(key) \
            .perform()

    def do_keystroke(self, key):
        ActionChains(self._driver) \
            .key_down(key) \
            .key_up(key) \
            .perform()

    def grab_screenshot(self, region, grayscale=True, filename=None):
        if filename == None:
            return pyautogui.screenshot(region=region)
        return pyautogui.screenshot(region=region, imageFilename=filename)

    def grab_screenshot_as_array(self, region):
        screenshot = self.grab_screenshot(region)
        data = np.asarray(screenshot, dtype=np.float32)
        data = data / 255
        data = np.mean(data, axis=2)
        return data

    def grab_game_screenshot_as_array(self):
        data = self.grab_screenshot_as_array(game_region)
        # data = data / 255
        return data

    def grab_score_screenshot_as_array(self):
        return self.grab_screenshot_as_array(score_region)

    def reset(self):
        time.sleep(1)
        self.do_keystroke(Keys.SPACE)
        time.sleep(1)
        self.do_action(2)

    def end(self):
        self._driver.close()