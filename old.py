


# def data_collection(agnt, env, policy):
#     replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#         data_spec=agnt.collect_data_spec,
#         batch_size=env.batch_size,
#         max_length=replay_buffer_capacity)

#     # policy = greedy_policy.GreedyPolicy(agent.collect_policy)
#     # agent.collect_policy = policy

#     collect_driver = dynamic_step_driver.DynamicStepDriver(
#         env,
#         policy,
#         observers=[replay_buffer.add_batch],
#         num_steps=collect_steps_per_iteration)

#     # Initial data collection
#     collect_driver.run()

#     # Dataset generates trajectories with shape [BxTx...] where
#     # T = n_step_update + 1.
#     dataset = replay_buffer.as_dataset(
#         num_parallel_calls=3, sample_batch_size=batch_size,
#         num_steps=2).prefetch(3)

#     iterator = iter(dataset)
#     return iterator, collect_driver, replay_buffer


# def train_one_iteration(agent, iterator, driver):

#     # Collect a few steps using collect_policy and save to the replay buffer.
#     driver.run()

#     # Sample a batch of data from the buffer and update the agent's network.
#     experience, unused_info = next(iterator)
#     train_loss = agent.train(experience)

#     iteration = agent.train_step_counter.numpy()
#     print('iteration: {0} loss: {1}'.format(iteration, train_loss.loss))


# game_io = IO()
# game_io.initialize()

# train_env, eval_env = set_up_env(game_io)
# global_step = tf.compat.v1.train.get_or_create_global_step()

# q_net = q_network.QNetwork(
#     train_env.observation_spec(),
#     train_env.action_spec(),
#     fc_layer_params=fc_layer_params)
# agent = set_up_agent(train_env, q_net, global_step)

# # q_plcy = q_policy.QPolicy(train_env.time_step_spec,
# #                             train_env.action_spec,
# #                             q_network=q_net)
# greedy_plcy = greedy_policy.GreedyPolicy(agent.collect_policy)

# iterator, collect_driver, replay_buffer = data_collection(
#     agent, train_env, greedy_plcy)

# agent.train = common.function(agent.train)


# checkpoint_dir = os.path.join(savedir, 'checkpoint')
# train_checkpointer = common.Checkpointer(
#     ckpt_dir=checkpoint_dir,
#     max_to_keep=1,
#     agent=agent,
#     policy=agent.policy,
#     replay_buffer=replay_buffer,
#     global_step=global_step
# )

# policy_dir = os.path.join(savedir, 'policy')
# tf_policy_saver = policy_saver.PolicySaver(agent.policy)


# def do_iterations(num, sets, total):
#     for i in range(num):
#         train_one_iteration(agent, iterator, collect_driver)
#         print('{}/{} iterations completed'.format(total+i+1, num*sets))

# # do_iterations(5)


# def run_episodes(policy, eval_tf_env, num_episodes=3):
#     for _ in range(num_episodes):
#         time_step = eval_tf_env.reset()
#         while not time_step.is_last():
#             action_step = policy.action(time_step)
#             time_step = eval_tf_env.step(action_step.action)


# def create_zip_file(dirname, base_filename):
#     return shutil.make_archive(base_filename, 'zip', dirname)


# def unzip_file_to(filepath, dirname):
#     with zipfile.ZipFile(filepath, 'r') as zip_ref:
#         zip_ref.extractall(dirname)


# print("saving checkpoint")
# train_checkpointer.save(global_step)
# checkpoint_zip_filename = create_zip_file(
#     checkpoint_dir, os.path.join(savedir, 'exported_cp'))

# print("loading checkpoint")
# unzip_file_to(checkpoint_zip_filename, checkpoint_dir)
# train_checkpointer.initialize_or_restore()
# global_step = tf.compat.v1.train.get_global_step()


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

# print("saving policy")
# tf_policy_saver.save(policy_dir)
# policy_zip_filename = create_zip_file(
#     policy_dir, os.path.join(savedir, 'exported_policy'))

# print("loading policy")
# unzip_file_to(policy_zip_filename, policy_dir)
# saved_policy = tf.compat.v2.saved_model.load(policy_dir)

# print("testing policy")
# run_episodes(saved_policy, eval_env)

# print("done!")
# game_io.end()
