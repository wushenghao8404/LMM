from resco_benchmark.agents.agent import Agent
from resco_benchmark.resco_config.signal_config import signal_configs
import numpy as np


class ActSequence(Agent):
    def __init__(self, act_space, map_name, ts_ids, n_step=6, step_length=10):
        super().__init__()
        self.full_act_space = [sub_act_space for i in range(n_step) for sub_act_space in act_space]
        self.act_space = act_space
        self.map_name = map_name
        self.ts_ids = ts_ids
        self.ts_id_to_index = {ts_id: i for i, ts_id in enumerate(self.ts_ids)}
        self.n_step = n_step
        self.n_traffic_signals = len(act_space)  # number of traffic signals
        self.step_length = step_length
        self.n_param = self.get_num_param()
        self.param = self.sample_param().reshape(self.n_step, self.n_traffic_signals)
        self.act_step = 0
        self.signal_config = signal_configs[self.map_name]
        self.neighbor_ids = []
        self.non_neighbor_ids = []
        for i, ts_id in enumerate(self.ts_ids):
            ts_neighbor_ids = []
            for key, neighbor_id in self.signal_config[ts_id]['downstream'].items():
                if neighbor_id is not None:
                    ts_neighbor_ids.append(self.ts_id_to_index[neighbor_id])
            self.neighbor_ids.append(ts_neighbor_ids)
            self.non_neighbor_ids.append(list(set(range(self.n_traffic_signals)) - set(ts_neighbor_ids + [i])))

    def local_perturb(self, param, scheme='FBLS1'):
        assert len(param) == self.n_param, f"Expect # param = {self.n_param}, but encounter with {len(param)}"
        solution = np.array(param).reshape(self.n_step, self.n_traffic_signals).astype(np.int_)
        if scheme == 'FBLS1':
            signal_id = np.random.randint(self.n_traffic_signals)
            for i in range(self.n_step):
                new_phase = np.random.randint(self.full_act_space[signal_id].n)
                while new_phase == solution[i, signal_id]:
                    new_phase = np.random.randint(self.full_act_space[signal_id].n)
                solution[i, signal_id] = new_phase
        elif scheme == 'FBLS2':
            signal_id = np.random.randint(self.n_traffic_signals)
            step_id = np.random.randint(self.n_step)
            new_phase = np.random.randint(self.full_act_space[signal_id].n)
            while new_phase == solution[step_id, signal_id]:
                new_phase = np.random.randint(self.full_act_space[signal_id].n)
            solution[step_id, signal_id] = new_phase
        elif scheme == 'FBLS3':
            signal_ids = [np.random.randint(self.n_traffic_signals)]
            if len(self.neighbor_ids[signal_ids[0]]) > 0:  # if the signal has a neighbor signal
                signal_ids.append(np.random.choice(self.neighbor_ids[signal_ids[0]]))
            for i in range(self.n_step):
                for signal_id in signal_ids:
                    new_phase = np.random.randint(self.full_act_space[signal_id].n)
                    while new_phase == solution[i, signal_id]:
                        new_phase = np.random.randint(self.full_act_space[signal_id].n)
                    solution[i, signal_id] = new_phase
        elif scheme == 'FBLS4':
            signal_ids = [np.random.randint(self.n_traffic_signals)]
            if len(self.neighbor_ids[signal_ids[0]]) > 0:  # if the signal has a neighbor signal
                signal_ids.append(np.random.choice(self.neighbor_ids[signal_ids[0]]))
            step_id = np.random.randint(self.n_step)
            for signal_id in signal_ids:
                new_phase = np.random.randint(self.full_act_space[signal_id].n)
                while new_phase == solution[step_id, signal_id]:
                    new_phase = np.random.randint(self.full_act_space[signal_id].n)
                solution[step_id, signal_id] = new_phase
        elif scheme == 'FBLS5':
            signal_ids = [np.random.randint(self.n_traffic_signals)]
            if len(self.non_neighbor_ids[signal_ids[0]]) > 0:  # if the signal has a non-neighbor signal
                signal_ids.append(np.random.choice(self.non_neighbor_ids[signal_ids[0]]))
            step_id = np.random.randint(self.n_step)
            for signal_id in signal_ids:
                new_phase = np.random.randint(self.full_act_space[signal_id].n)
                while new_phase == solution[step_id, signal_id]:
                    new_phase = np.random.randint(self.full_act_space[signal_id].n)
                solution[step_id, signal_id] = new_phase
        else:
            raise ValueError
        return solution

    def sample_param(self):
        param = [sub_act_space.sample() for sub_act_space in self.full_act_space]
        return np.array(param)

    def get_num_param(self):
        return int(self.n_step * self.n_traffic_signals)

    def load_param(self, param):
        # print(self.n_param, len(param))
        assert len(param) == self.n_param, f"Expect # param = {self.n_param}, but encounter with {len(param)}"
        self.param = np.array(param).reshape(self.n_step, self.n_traffic_signals).astype(np.int_)

    def act(self, observation):
        acts = dict()
        decoded_act = self.param[self.act_step]
        # print(observation.keys())
        # print(self.ts_id_to_index)
        # print(self.neighbor_ids)
        # print(self.non_neighbor_ids)
        # neighbours = []

        for i, agent_id in enumerate(observation.keys()):
            acts[agent_id] = decoded_act[i]
            # neighbours.append(self.signal_config[agent_id]['downstream'])
        # print(neighbours)
        self.act_step = (self.act_step + 1) % self.n_step
        return acts

    def observe(self, observation, reward, done, info):
        pass