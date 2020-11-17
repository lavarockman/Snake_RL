from game_IO import IO
from snake_env import SnakeEnv
from main import set_up_env

import tensorflow as tf

policy_dir = r"C:\Users\Levi\Programming\Python\Deep_Learning\Snake\saves\policy"


def run_episodes(policy, eval_tf_env):
    num_episodes = 3
    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)


io = IO()
io.initialize()

train_env, eval_env = set_up_env(io)

saved_policy = tf.compat.v2.saved_model.load(policy_dir)
# saved_policy = tf.compat.v2.saved_model.load("")
run_episodes(saved_policy, eval_env)

io.end()