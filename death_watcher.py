import numpy as np
import pyautogui as gui

x = 831
y = 251
match_pixel = (74, 117, 44)

class Death_Watcher:

    def __init__(self, io):
        self._io = io
        self.reset()

    def reset(self):
        pass

    def is_dead(self):
        return not gui.pixelMatchesColor(x, y, match_pixel)
