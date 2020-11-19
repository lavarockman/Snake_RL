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
import configparser

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
config = configparser.ConfigParser()
model_config = config['Model']
training_config = config['Training']

model_name = model_config['model_name']
batch_size = model_config['batch_size']
collect_steps_per_iteration = model_config['collect_steps_per_iteration']
replay_buffer_max_length = model_config['replay_buffer_max_length']
log_interval = model_config['log_interval']
num_eval_episodes = model_config['num_eval_episodes']
eval_interval = model_config['eval_interval']

num_iteraions = training_config['num_iteraions']
num_checkpoints = training_config['num_checkpoints']
learning_rate = training_config['learning_rate']
epsilon_start = training_config['epsilon_starting_num']
epsilon_decay = training_config['epslion_decay_rate']

epsilon = epsilon_start
fc_layer_params = (100,)
savedir = os.path.join(os.getcwd(), 'models', model_name)


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
        train_step_counter=global_step,
        debug_summaries=True)
    agent.initialize()
    return agent


def set_up_saver(agent, replay_buffer):
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
    return train_checkpointer, tf_policy_saver, checkpoint_dir, policy_dir


def get_policy(env, q_net, epsilon_callback):

    q_plcy = q_policy.QPolicy(env.time_step_spec(),
                              env.action_spec(),
                              q_network=q_net)
    # greedy_plcy = greedy_policy.GreedyPolicy(q_plcy)
    ep_greedy_plcy = epsilon_greedy_policy.EpsilonGreedyPolicy(
        q_plcy, epsilon_callback)
    plcy = ep_greedy_plcy
    return plcy


def data_collection(agnt, env, policy):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agnt.collect_data_spec,
        batch_size=env.batch_size,
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


def do_iterations(num, total, current):
    for i in range(num):
        train_one_iteration(agent, iterator, collect_driver)
        print('{}/{} iterations completed'.format(current+i+1, total))


def do_checkpoints(total, checkpoints, saver):
    reps = math.floor(total / checkpoints)
    count = 0
    print('running {} iterations with {} checkpoints...'.format(total, checkpoints))
    for _ in range(checkpoints + 1):
        if count >= total:
            break
        iterations = reps
        if iterations + count > total:
            iterations = total - count
        print("doing {} iterations".format(iterations))
        do_iterations(iterations, total, count)
        count += iterations
        print("{}/{} iterations completed".format(count, total))
        saver.save()


def run_episodes(policy, eval_tf_env, num_episodes=3):
    for _ in range(num_episodes):
        time_step = eval_tf_env.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)

def epsilon_decrease():
    epsilon *= epsilon_decay
    return epsilon

game_io = IO()
print('initializing game io')
game_io.initialize()

print('setting up environment')
train_env, eval_env = set_up_env(game_io)
global_step = tf.compat.v1.train.get_or_create_global_step()

print('creating q net')
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

print('setting up agent')
agent = set_up_agent(train_env, q_net, global_step)
print('getting policy')
policy = get_policy(train_env, q_net, epsilon_decrease)
agent.train = common.function(agent.train)

print('initial data collection')
iterator, collect_driver, replay_buffer = data_collection(
    agent, train_env, policy)

print('setting up checkpoint and policy saver')
checkpoint_saver, tf_policy_saver, checkpoint_dir, policy_dir = set_up_saver(
    agent, replay_buffer)

print("initializing or restoring checkpoint")
checkpoint_saver.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()

print('running iterations')
do_checkpoints(num_iteraions, num_checkpoints, checkpoint_saver)

print("saving policy")
tf_policy_saver.save(policy_dir)

print("loading policy")
saved_policy = tf.compat.v2.saved_model.load(policy_dir)

print("testing policy")
run_episodes(saved_policy, eval_env)

print("done!")
game_io.end()
