import numpy as np
from resco_benchmark.agents.agent import SharedAgent, Agent
from resco_benchmark.resco_config.signal_config import signal_configs
from copy import deepcopy
import gym


class LinearNN:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(self.input_dim, self.output_dim)
        self.b = np.random.randn(1, self.output_dim)
        self.param_shape_dict = {'W': self.W.shape,
                                 'b': self.b.shape}
        self.n_param = self.get_num_param()

    def get_num_param(self):
        return (self.input_dim + 1) * self.output_dim

    def load_param(self, param):
        assert len(param) == self.n_param
        index = 0
        for key, value in self.param_shape_dict.items():
            param_size = int(np.prod(self.param_shape_dict[key]))
            param_to_copy = param[index:index + param_size].reshape(self.param_shape_dict[key])
            setattr(self, key, param_to_copy)
            index += param_size

    def forward(self, obs):
        X = np.atleast_2d(np.array(deepcopy(obs)))
        return X @ self.W + self.b


class SimpleNN:
    def __init__(self, input_dim, output_dim, hidden_dim=8):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim)
        self.b1 = np.random.randn(1, self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim)
        self.b2 = np.random.randn(1, self.output_dim)
        self.param_shape_dict = {'W1': self.W1.shape,
                                 'b1': self.b1.shape,
                                 'W2': self.W2.shape,
                                 'b2': self.b2.shape}
        self.n_param = self.get_num_param()

    def get_num_param(self):
        return (self.input_dim + 1) * self.hidden_dim + (self.hidden_dim + 1) * self.output_dim

    def load_param(self, param):
        assert len(param) == self.n_param
        index = 0
        for key, value in self.param_shape_dict.items():
            param_size = int(np.prod(self.param_shape_dict[key]))
            param_to_copy = param[index:index + param_size].reshape(self.param_shape_dict[key])
            setattr(self, key, param_to_copy)
            index += param_size

    def forward(self, obs):
        X = np.atleast_2d(np.array(deepcopy(obs)))
        return np.maximum(0, X @ self.W1 + self.b1) @ self.W2 + self.b2


class NeuralUrgency(SharedAgent):
    def __init__(self, config, obs_act, map_name, thread_number, param=None):
        super().__init__(config, obs_act, map_name, thread_number)
        self.valid_acts = signal_configs[map_name]['valid_acts']
        self.max_lanes = signal_configs[map_name]['max_lanes']
        self.agent = NeuralUrgencyAgent(signal_configs[map_name]['phase_pairs'], self.max_lanes, param=param)
        # print('input_param', param)

    def get_solution_space(self):
        return gym.spaces.Box(low=-5, high=5, shape=(self.agent.nn.get_num_param(),))

    def sample_param(self):
        return self.get_solution_space().sample()


class NeuralUrgencyAgent(Agent):
    def __init__(self, phase_pairs, max_lanes, param=None):
        super().__init__()
        self.phase_pairs = phase_pairs
        self.max_lanes = max_lanes
        # self.nn = LinearNN(self.max_lanes * 2 + 1, 1)
        self.nn = SimpleNN(self.max_lanes * 2 + 1, 1)
        if param is not None:
            self.nn.load_param(param)

    def teach(self, observations, valid_acts=None, reverse_valid=None):
        acts = []
        for i, observation in enumerate(observations):
            if valid_acts is None:
                state_values = []
                for pair in self.phase_pairs:
                    x = [observation[0]]
                    x += observation[pair[0] + 1]
                    x += observation[pair[1] + 1]
                    x += [0] * (self.nn.input_dim - len(x))
                    state_values.append(np.sum(x))
                acts.append(np.argmax(state_values))
            else:
                max_value, max_index = None, None
                for idx in valid_acts[i]:
                    pair = self.phase_pairs[idx]
                    x = [observation[0]]
                    x += observation[pair[0] + 1]
                    x += observation[pair[1] + 1]
                    x += [0] * (self.nn.input_dim - len(x))
                    state_val = np.sum(x)
                    if max_value is None:
                        max_value = state_val
                        max_index = idx
                    if state_val > max_value:
                        max_value = state_val
                        max_index = idx
                acts.append(valid_acts[i][max_index])
        return acts

    def act(self, observations, valid_acts=None, reverse_valid=None):
        acts = []
        for i, observation in enumerate(observations):
            if valid_acts is None:
                state_values = []
                for pair in self.phase_pairs:
                    x = [observation[0]]
                    x += observation[pair[0] + 1]
                    x += observation[pair[1] + 1]
                    x += [0] * (self.nn.input_dim - len(x))
                    x = np.array(x)[None, :]
                    state_values.append(self.nn.forward(x).flatten()[0])
                acts.append(np.argmax(state_values))
            else:
                max_value, max_index = None, None
                for idx in valid_acts[i]:
                    pair = self.phase_pairs[idx]
                    x = [observation[0]]
                    x += observation[pair[0] + 1]
                    x += observation[pair[1] + 1]
                    x += [0] * (self.nn.input_dim - len(x))
                    x = np.array(x)[None, :]
                    state_val = self.nn.forward(x).flatten()[0]
                    if max_value is None:
                        max_value = state_val
                        max_index = idx
                    if state_val > max_value:
                        max_value = state_val
                        max_index = idx
                acts.append(valid_acts[i][max_index])
        return acts

    def observe(self, observation, reward, done, info):
        pass

    def save(self, path):
        pass


class NeuralUrgency_v2(SharedAgent):
    def __init__(self, config, obs_act, map_name, thread_number, param=None):
        super().__init__(config, obs_act, map_name, thread_number)
        self.valid_acts = signal_configs[map_name]['valid_acts']
        self.max_lanes = signal_configs[map_name]['max_lanes']
        self.agent = NeuralUrgencyAgent_v2(signal_configs[map_name]['phase_pairs'], self.max_lanes, param=param)
        # print('input_param', param)

    def get_solution_space(self):
        return gym.spaces.Box(low=-5, high=5, shape=(self.agent.nn.get_num_param(),))

    def sample_param(self):
        return self.get_solution_space().sample()


class NeuralUrgencyAgent_v2(Agent):
    def __init__(self, phase_pairs, max_lanes, param=None):
        super().__init__()
        self.phase_pairs = phase_pairs
        self.max_lanes = max_lanes
        self.nn = SimpleNN(self.max_lanes * 3 + 1, 1)
        # self.nn = LinearNN(1, 1)
        if param is not None:
            self.nn.load_param(param)

    def act(self, observations, valid_acts=None, reverse_valid=None):
        acts = []
        for i, observation in enumerate(observations):
            if valid_acts is None:
                state_values = []
                for pair in self.phase_pairs:
                    x = [observation[0]]
                    x += observation[pair[0] + 1]
                    x += observation[pair[1] + 1]
                    x += [0] * (self.nn.input_dim - len(x))
                    x = np.array(x)[None, :]
                    state_values.append(self.nn.forward(x).flatten()[0])
                acts.append(np.argmax(state_values))
            else:
                max_value, max_index = None, None
                for idx in valid_acts[i]:
                    pair = self.phase_pairs[idx]
                    x = [observation[0]]
                    x += observation[pair[0] + 1]
                    x += observation[pair[1] + 1]
                    x += [0] * (self.nn.input_dim - len(x))
                    x = np.array(x)[None, :]
                    state_val = self.nn.forward(x).flatten()[0]
                    if max_value is None:
                        max_value = state_val
                        max_index = idx
                    if state_val > max_value:
                        max_value = state_val
                        max_index = idx
                acts.append(valid_acts[i][max_index])
        return acts

    def observe(self, observation, reward, done, info):
        pass

    def save(self, path):
        pass