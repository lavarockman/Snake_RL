from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts
from queue import Empty

from score_keeper import Score_Keeper
from death_watcher import Death_Watcher

shape = (748, 746)
exist_reward = 0.01
apple_reward = 1
death_reward = -10

class SnakeEnv(py_environment.PyEnvironment):

  def __init__(self, io):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
    # 0=left; 1=up; 2=right; 3=down; 4=do nothing;

    self._observation_spec = array_spec.BoundedArraySpec(
        shape=shape, dtype=np.float32, minimum=0.0, maximum=1.0, name='observation')

    self._episode_ended = False
    self._io = io
    self._score_keeper = Score_Keeper(io)
    self._death = Death_Watcher(io)
  
  def action_spec(self):
    return self._action_spec
  
  def observation_spec(self):
    return self._observation_spec
  
  def _reset(self):
    self._io.reset()
    self._score_keeper.reset()
    self._episode_ended = False
    return ts.restart(self._io.grab_game_screenshot_as_array())
  
  def _step(self, action):

    if self._episode_ended:
      return self.reset()
    
    self._io.do_action(action)

    reward = exist_reward
    
    if(self._score_keeper.did_score_change()):
      reward += apple_reward
    
    if self._death.is_dead():
      reward += death_reward
      self._episode_ended = True
    
    
    if self._episode_ended:
      return ts.termination(self._io.grab_game_screenshot_as_array(), reward)
    else:
      return ts.transition(self._io.grab_game_screenshot_as_array(), reward=reward, discount=1.0)