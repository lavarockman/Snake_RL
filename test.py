from game_IO import IO
from death_watcher import Death_Watcher
import time
import pyautogui as gui

io = IO()
io.initialize()

data = io.grab_game_screenshot_as_array()
print(data.shape)

# d = Death_Watcher()
# d.initialize(io)

# print(d.is_dead())
# io.reset()
# print(d.is_dead())
# time.sleep(5)
# print(d.is_dead())

# io.end()
# time.sleep(1)
# print(d.is_dead())

# region = (831, 251, 1, 1)
# x = 831
# y = 251

# io.reset()
# pix = gui.pixel(x, y)
# print(pix)

# io.end()