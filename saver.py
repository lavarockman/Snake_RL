import os
import shutil
import zipfile

from tf_agents.utils import common
from tf_agents.policies import policy_saver


class Saver():

    def __init__(self, agent, replay_buffer, global_step, save_dir):
        self._agent = agent
        self._save_dir = save_dir
        self._replay_buffer = replay_buffer
        self._global_step = global_step

    def initiate(self):
        self._checkpoint_dir = os.path.join(self._save_dir, 'checkpoint')
        self._train_checkpointer = common.Checkpointer(
            ckpt_dir=self._checkpoint_dir,
            max_to_keep=1,
            agent=self._agent,
            policy=self._agent.policy,
            replay_buffer=self._replay_buffer,
            global_step=self._global_step
        )

        self._policy_dir = os.path.join(self._save_dir, 'policy')
        self._tf_policy_saver = policy_saver.PolicySaver(self._agent.policy)

    def save_checkpoint(self, zip=False):
        print("saving checkpoint")
        self._train_checkpointer.save(self._global_step)
        print("saved!")

