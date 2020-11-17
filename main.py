from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import multiprocessing as mp
import time
import os
import zipfile
import shutil
import tempfile
import io
import math

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import q_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import dynamic_step_driver

from selenium.webdriver.common.keys import Keys

from game_IO import IO
from score_keeper import Score_Keeper
from death_watcher import Death_Watcher
from snake_env import SnakeEnv

tf.compat.v1.enable_v2_behavior()

relative_path = "models/ep_greedy"
# relative_path = "models/normal"
savedir = os.path.join(os.getcwd(), relative_path)
checkpoint_zip_filename = r"C:\Users\Levi\Programming\Python\Deep_Learning\Snake\models\ep_greedy\exported_cp.zip"

collect_steps_per_iteration = 100
replay_buffer_max_length = 1000

fc_layer_params = (100,)

batch_size = 32
learning_rate = 1e-3
log_interval = 5

num_eval_episodes = 10
eval_interval = 25


def set_up_env(io):
    env_py = SnakeEnv(io)
    train_env = tf_py_environment.TFPyEnvironment(env_py)
    eval_env = tf_py_environment.TFPyEnvironment(env_py)
    return train_env, eval_env


def set_up_agent(env, q_net, global_step):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_step)
    agent.initialize()
    return agent


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


def data_collection(agnt, env, policy):
    # replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    #     data_spec=agnt.collect_data_spec,
    #     batch_size=env.batch_size,
    #     max_length=replay_buffer_capacity)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        env,
        policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration)

    # Initial data collection
    collect_driver.run()

    # Dataset generates trajectories with shape [BxTx...] where
    # T = n_step_update + 1.
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)
    return iterator, collect_driver, replay_buffer


def train_one_iteration(agent, iterator, driver):

    # Collect a few steps using collect_policy and save to the replay buffer.
    driver.run()

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience)

    iteration = agent.train_step_counter.numpy()
    print('iteration: {0} loss: {1}'.format(iteration, train_loss.loss))


game_io = IO()
game_io.initialize()

train_env, eval_env = set_up_env(game_io)
global_step = tf.compat.v1.train.get_or_create_global_step()

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)
agent = set_up_agent(train_env, q_net, global_step)

q_plcy = q_policy.QPolicy(train_env.time_step_spec(),
                          train_env.action_spec(),
                          q_network=q_net)

greedy_plcy = greedy_policy.GreedyPolicy(q_plcy)
ep_greedy_plcy = epsilon_greedy_policy.EpsilonGreedyPolicy(q_plcy, 1.0)
plcy = ep_greedy_plcy

# print("overriding greedy policy")
# greedy_plcy = agent.collect_policy

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
# avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
# returns = [avg_return]
returns = []

checkpoint_dir = os.path.join(savedir, 'checkpoint')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)

policy_dir = os.path.join(savedir, 'policy')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

def do_iterations(num, returns):
    for _ in range(num):
        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(train_env, plcy, replay_buffer, collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            print('evaluating')
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)


def save_checkpoint():
    print("saving checkpoint")
    train_checkpointer.save(global_step)
    checkpoint_zip_filename = create_zip_file(
        checkpoint_dir, os.path.join(savedir, 'exported_cp'))
    print("saved!")
    return checkpoint_zip_filename

def do_checkpoints(total, checkpoints, returns):
    reps = math.floor(total / checkpoints)
    count = 0
    for _ in range(checkpoints + 1):
        if count >= total:
            break
        iterations = reps
        if iterations + count > total:
            iterations = total - count
        print("doing {} iterations".format(iterations))
        do_iterations(iterations, returns)
        count += iterations
        print("{}/{} iterations completed".format(count, total))
        # print("completed iterations, saving checkpoint")
        save_checkpoint()
        



def run_episodes(policy, eval_tf_env, num_episodes=3):
    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)


def create_zip_file(dirname, base_filename):
    return shutil.make_archive(base_filename, 'zip', dirname)


def unzip_file_to(filepath, dirname):
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(dirname)


checkpoint_zip_filename = save_checkpoint()

print("loading checkpoint")
unzip_file_to(checkpoint_zip_filename, checkpoint_dir)
train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()


do_checkpoints(200, 10, returns)

# def run_iterations_with_checkpoints(reps, sets):
#     total = 1
#     for i in range(sets):
#         print("set {}/{}".format(i+1, sets))
#         print("training reps")
#         do_iterations(reps, sets, total)
#         total += reps

#         print("saving checkpoint")
#         train_checkpointer.save(global_step)
#         create_zip_file(checkpoint_dir, os.path.join(savedir, 'exported_cp'))
#         print("saved!")


# run_iterations_with_checkpoints(15, 2)

print("saving policy")
tf_policy_saver.save(policy_dir)
policy_zip_filename = create_zip_file(
    policy_dir, os.path.join(savedir, 'exported_policy'))

print("loading policy")
unzip_file_to(policy_zip_filename, policy_dir)
saved_policy = tf.compat.v2.saved_model.load(policy_dir)

print("testing policy")
run_episodes(saved_policy, eval_env)

print("done!")
game_io.end()
