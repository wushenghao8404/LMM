from resco_benchmark.agents.agent import Agent
import numpy as np
import pickle
import config
import os


class FixedTiming(Agent):
    def __init__(self, scene_name, case, tau):
        super().__init__()
        self.scene_name = scene_name
        self.case = case
        self.tau = tau
        with open(os.path.join(config.PROJECT_PATH, 'data', 'mdl', self.scene_name, f'c{self.case}',
                               'lmm_r0.pkl'), 'rb') as f:
            self.mdl = pickle.load(f)
        self.x = self.mdl.map(self.tau)

    def act(self, observation):
        acts = dict()
        for i, agent_id in enumerate(observation.keys()):
            acts[agent_id] = 0
        return acts

    def observe(self, observation, reward, done, info):
        pass