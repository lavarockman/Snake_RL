import numpy as np
import multiprocessing as mp
from queue import Empty


class Score_Keeper:
    def __init__(self, io):
        self._io = io

    def reset(self):
        self._last_pic = self._io.grab_score_screenshot_as_array()

    def did_score_change(self):
        next_pic = self._io.grab_score_screenshot_as_array()
        changed = False
        if not np.array_equal(next_pic, self._last_pic):
            changed = True
            self._last_pic = next_pic
        return changed
