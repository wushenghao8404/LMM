import numpy as np
import config
import utils
import random
import os
import shutil
import dill
from resco_benchmark.multi_signal import MultiSignal
from resco_benchmark.resco_config.agent_config import agent_configs
from resco_benchmark.resco_config.map_config import map_configs
import zlib
from copy import deepcopy


class Evaluator:
    def __init__(self, name, use_dec=False):
        self.name = name
        self.use_dec = use_dec
        self.nfe = 0

    def init(self):
        raise NotImplementedError

    def evaluate(self, prob, x):
        raise NotImplementedError


class Problem:
    def __init__(self, dim, xlb, xub):
        self.dim = dim
        if len(xlb) != self.dim or len(xub) != self.dim:
            raise Exception("Unmatched size of boundary array")
        self.xlb = xlb
        self.xub = xub

    def decode(self, geno_x):
        # translate from a genotype to a phenotype, e.g., weight parameters of the neural net-> struct or object of
        # the neural net, the solution validation such as boundary handling can also be implemented in this
        pass

    def encode(self, pheno_x):
        pass

    def func(self, x):
        pass
    
    def func_with_worker(self, x_workerid):
        pass


class TSTOProblem(Problem):
    """Traffic Signal Timing Problem taking duration/offset as optimization variable"""
    def __init__(self, scene_name, case, net_input, simulation_id=None):
        self.scene_name = scene_name
        self.case = case
        self.cfg = config.config_dictionary[self.scene_name]
        super().__init__(self.get_dim_sol(), self.get_xlb(), self.get_xub())
        assert (len(net_input) == self.get_dim_prob())
        self.net_input = np.array(net_input, dtype=np.int)
        self.simulation_id = simulation_id
        # prepare net file
        self.net_path = utils.get_net_path(self.scene_name)

    def prepare(self):
        self.sim_dir = utils.get_sim_cfg_dir_path(scene_name=self.scene_name, case=self.case,
                                                  simulation_id=self.simulation_id, worker_id=None)
        if not os.path.exists(self.sim_dir):
            os.makedirs(self.sim_dir)
        # prepare add file
        self.prepare_add_file()
        # prepare cfg file
        self.prepare_cfg_file()
        # prepare route file
        self.rou_path = utils.get_rou_path(scene_name=self.scene_name, case=self.case, net_input=self.net_input)
        # prepare output file
        self.output_path = os.path.join(self.sim_dir, 'output.xml')

    def prepare_worker(self, worker_id):
        self.sim_dir = utils.get_sim_cfg_dir_path(scene_name=self.scene_name, case=self.case,
                                                  simulation_id=self.simulation_id, worker_id=worker_id)
        if not os.path.exists(self.sim_dir):
            os.makedirs(self.sim_dir)
        # prepare add file
        self.prepare_add_file()
        # prepare cfg file
        self.prepare_cfg_file()
        # prepare route file
        self.rou_path = utils.get_rou_path(scene_name=self.scene_name, case=self.case, net_input=self.net_input)
        # prepare output file
        self.output_path = os.path.join(self.sim_dir, 'output.xml')

    def prepare_add_file(self):
        self.add_path = os.path.join(self.sim_dir, self.scene_name + '.add.xml')
        if not os.path.exists(self.add_path):
            print(f"copying {utils.get_template_add_path(self.scene_name)} to {self.add_path}")
            shutil.copyfile(utils.get_template_add_path(self.scene_name), self.add_path)

    def prepare_cfg_file(self):
        self.cfg_path = os.path.join(self.sim_dir, self.scene_name + '.sumocfg')
        if not os.path.exists(self.cfg_path):
            print(f"copying {utils.get_template_cfg_path()} to {self.cfg_path}")
            shutil.copyfile(utils.get_template_cfg_path(), self.cfg_path)

    def get_xlb(self):
        return self.cfg.get_xlb()

    def get_xub(self):
        return self.cfg.get_xub()
    
    def sample_x(self, nx=1):
        if nx == 1:
            return np.random.uniform(self.xlb, self.xub).astype(np.int)
        else:
            return np.random.uniform(self.xlb, self.xub, (nx, self.dim)).astype(np.int)
    
    def get_dim_prob(self):
        return self.cfg.DIM_PROB_FEATURE

    def get_dim_sol(self):
        return self.cfg.DIM_DECVAR

    def get_net_input(self):
        return self.net_input

    def func(self, x):
        x0 = np.array(np.clip(x, self.get_xlb(), self.get_xub()), dtype=np.int)
        # prepare all file paths necessary for simulation
        self.prepare()
        # load route file into '.sumocfg'
        utils.load_cfg_file(self.cfg_path, self.net_path, self.rou_path, self.add_path, self.output_path)
        # load additional file (traffic signal plan) into '.sumocfg'
        utils.load_traffic_light_program(x0, self.cfg, self.add_path)
        # run a sumo simulation process
        utils.run_simulation_cmd(self.cfg_path)
        # read the results from outputs
        res = utils.read_simulation_output(self.output_path)
        return res[1]

    def func_with_worker(self, x_workerid):
        x, workerid = x_workerid[0], x_workerid[1]
        x0 = np.array(np.clip(x, self.get_xlb(), self.get_xub()), dtype=np.int)
        # prepare all file paths necessary for simulation
        self.prepare_worker(worker_id=workerid)
        # load route file into '.sumocfg'
        utils.load_cfg_file(self.cfg_path, self.net_path, self.rou_path, self.add_path, self.output_path)
        # load additional file (traffic signal plan) into '.sumocfg'
        utils.load_traffic_light_program(x0, self.cfg, self.add_path)
        # run a sumo simulation
        utils.run_simulation_cmd(self.cfg_path)
        # read the results from outputs
        res = utils.read_simulation_output(self.output_path)
        return res[1]


class TSTOProblemGenerator:
    def __init__(self, scene_name, case, rnd_seed=None, simulation_id=None):
        self.scene_name = scene_name
        self.case = case
        self.cfg = config.config_dictionary[self.scene_name]
        self.tdim = self.cfg.DIM_PROB_FEATURE
        self.tlb = self.cfg.get_tlb()
        self.tub = self.cfg.get_tub()
        self.xdim = self.cfg.DIM_DECVAR
        self.xlb = self.cfg.get_xlb()
        self.xub = self.cfg.get_xub()
        self.simulation_id = simulation_id
        if self.simulation_id is None:
            # self.simulation_id = utils.generate_unique_id()
            print(f'Assigned unique simulation id: {self.simulation_id}')
        else:
            print(f'Assigned unique simulation id: {self.simulation_id}')
        if isinstance(rnd_seed, int):
            random.seed(rnd_seed)
            np.random.seed(rnd_seed)

    def get_tlb(self):
        return self.cfg.get_tlb()

    def get_tub(self):
        return self.cfg.get_tub()

    def sample_prob_instance(self, sample_func=None):
        if not sample_func:
            raise ValueError
            tau = np.random.uniform(self.tlb, self.tub)
            prob = TSTOProblem(self.scene_name, self.case, tau, simulation_id=self.simulation_id)
            return prob
        else:
            tau = sample_func()
            prob = TSTOProblem(self.scene_name, self.case, tau, simulation_id=self.simulation_id)
            return prob

    def get_prob_instance(self, tau):
        prob = TSTOProblem(self.scene_name, self.case, tau, simulation_id=self.simulation_id)
        return prob


class TrafficPerformanceEvaluator(Evaluator):
    def __init__(self, mode, n_workers=1):
        super().__init__("TrafficPerformanceEvaluator")
        if mode not in ['spsi', 'spmi', 'mpmi']:
            raise Exception('Unexpected TLC evaluator mode')
        self.mode = mode
        self.n_workers = n_workers
        self.workerids = [str(i) for i in range(self.n_workers)]

    def evaluate_with_worker(self, prob_x_workerid):
        prob, x, workerid = prob_x_workerid[0], prob_x_workerid[1], prob_x_workerid[2]
        return prob.func_with_worker((x, workerid))

    def evaluate_without_worker(self, prob_x):
        prob, x = prob_x[0], prob_x[1]
        return prob.func(x)

    def evaluate(self, p, x, pool=None):
        if self.mode == 'spsi':
            if not isinstance(p, Problem):
                raise Exception("Unexpected input under spsi mode. Expected a problem instance")
            return self.spsi_evaluate(p, x)
        elif self.mode == 'spmi':
            if not isinstance(p, Problem):
                raise Exception("Unexpected input under spmi mode. Expected a problem instance")
            return self.spmi_evaluate(p, x, pool)
        elif self.mode == 'mpmi':
            if not isinstance(p, list):
                raise Exception("Unexpected input under mpmi mode. Expected a list containing problem instances")
            return self.mpmi_evaluate(p, x, pool)
        else:
            raise Exception("Unexpected evaluation mode")

    def spsi_evaluate(self, prob, x):
        """
            single problem single input evaluation
        """
        if x.ndim != 1:
            raise Exception("The input X must be 1d")
        self.nfe += 1
        return self.evaluate_with_worker((prob, x, self.workerids[0]))
    
    def spmi_evaluate(self, prob, X, pool):
        """
            single problem multiple input parallel evaluation
        """
        if X.ndim != 2:
            raise Exception("The input X must be 2d")
        # evaluate batch solutions on single problem
        X_List = list(X)
        Y_List = []
        self.nfe += len(X_List)
        if self.n_workers > 1:
            assert pool is not None
            for i in range(len(X_List) // self.n_workers + 1):
                tX_List = X_List[i * self.n_workers:(i + 1) * self.n_workers]
                Y_List += pool.map(self.evaluate_with_worker,
                                   zip([prob] * len(tX_List), tX_List, self.workerids[:len(tX_List)]))
            return np.array(Y_List)
        else:
            return np.fromiter(map(lambda x: self.evaluate_without_worker((x[0], x[1])), zip([prob] * len(X_List), X_List)),
                               dtype=np.float)

    def mpmi_evaluate(self, prob_List, X, pool):
        """
            multiple problem multiple input parallel evaluation
        """
        if X.ndim != 2:
            raise Exception("The input X must be 2d")
        nX = X.shape[0]
        if nX != len(prob_List):
            raise Exception("The problem amount and the solution amount do not match")
        # evaluate batch solutions on multiple problems
        X_List = list(X)
        Y_List = []
        self.nfe += len(X_List)
        if self.n_workers > 1:
            assert pool is not None
            for i in range(len(X_List) // self.n_workers + 1):
                tX_List = X_List[i * self.n_workers:(i + 1) * self.n_workers]
                tprob_List = prob_List[i * self.n_workers:(i + 1) * self.n_workers]
                Y_List += pool.map(self.evaluate_with_worker, zip(tprob_List, tX_List, self.workerids[:len(tX_List)]))
            return np.array(Y_List)
        else:
            return np.fromiter(map(lambda x: self.evaluate_without_worker((x[0], x[1])), zip(prob_List, X_List)), dtype=np.float)


class TSCProblem(Problem):
    """Traffic Signal Timing Problem taking action sequence as optimization variable"""
    def __init__(self, scene_name, case, agent_name, net_input, penalty=True, verbose=False, reward_as_fitness=False):
        self.scene_name = scene_name
        self.case = case
        self.cfg = config.config_dictionary[self.scene_name]
        self.n_step = 90
        self.step_length = 10
        assert (len(net_input) == len(self.cfg.get_tlb()))
        self.agent_name = agent_name
        self.tau = net_input
        self.penalty = penalty
        self.penalty_factor = 1.0
        self.map_config = map_configs[scene_name]
        self.verbose = verbose
        self.connection_name = self.scene_name + '-c' + str(case) + '-' + agent_name
        self.reward_as_fitness = reward_as_fitness

        if self.agent_name == "NeuralUrgency" and scene_name == "ingolstadt7":
            # tune this hyperparameter for NeuralUrgency
            self.penalty_factor = 0.1
        else:
            pass

        if self.agent_name in ["TinyAgent", "NeuralWave", "NeuralUrgency", "NeuralUrgency_v2"]:
            agt, env = get_agent_and_env(scene_name, case, agent_name, net_input,
                                         worker_id=string_to_worker_id(self.connection_name))
            sol_space = agt.get_solution_space()
            super().__init__(sol_space.shape[0], xlb=np.ones(sol_space.shape[0]) * sol_space.low,
                             xub=np.ones(sol_space.shape[0]) * sol_space.high)
            # print('sol space', sol_space)

        elif self.agent_name in ["ActSequence", "Adapter"]:
            agt, env = get_agent_and_env(scene_name, case, "ActSequence", net_input,
                                         worker_id=string_to_worker_id(self.connection_name))
            super().__init__(agt.n_param, xlb=np.zeros(agt.n_param),
                             xub=np.array([sub_act_space.n for sub_act_space in agt.full_act_space]))
            self.n_ts = agt.n_traffic_signals
            self.act_space = agt.act_space
            if self.agent_name == "ActSequence":
                self.n_step = agt.n_step
                self.n_param = agt.n_param
                self.neighbor_ids = agt.neighbor_ids
                self.non_neighbor_ids = agt.non_neighbor_ids
                self.ts_ids = agt.ts_ids
                self.full_act_space = agt.full_act_space

    def get_xlb(self):
        return self.xlb

    def get_xub(self):
        return self.xub

    def get_dim_sol(self):
        return self.dim

    def sample_x(self):
        return np.random.uniform(self.xlb, self.xub)

    def func_with_worker(self, x_workerid):
        x, worker_id = x_workerid[0], x_workerid[1]
        if self.agent_name == "Adapter":
            adapt_acts = np.array(x).astype(np.int_).reshape(self.n_step, self.n_ts)
            base_agt, env = get_agent_and_env(self.scene_name, self.case, "IDQN", self.tau, trial=0,
                                              worker_id=string_to_worker_id(self.connection_name + f"-w{worker_id}"))
        else:
            agent, env = get_agent_and_env(self.scene_name, self.case, self.agent_name, self.tau, trial=worker_id,
                                           param=np.clip(x, self.xlb, self.xub),
                                           worker_id=string_to_worker_id(self.connection_name + f"-w{worker_id}"))
        obs = env.reset()
        done = False
        total_rew = 0.
        step = 0
        while not done:
            if self.agent_name == "Adapter":
                base_act = base_agt.act(obs)
                act = deepcopy(base_act)
                for i, agent_id in enumerate(obs.keys()):
                    act[agent_id] = (act[agent_id] + adapt_acts[step, i]) % self.act_space[i].n
            else:
                act = agent.act(obs)
            obs, rew, done, info = env.step(act)
            total_rew += sum(rew.values())
            step += 1
        env.close()
        # print(act_seq)
        # print(env.redirected_output_path)

        if self.reward_as_fitness:
            fitness = -total_rew
            if self.verbose:
                print(f"fit={fitness:.2f}")
        else:
            n_veh_finish, avg_delay = utils.read_simulation_output(outputPATH=env.redirected_output_path, verbose=0)
            fitness = avg_delay + (np.sum(self.tau) - n_veh_finish) * self.penalty * self.penalty_factor
            if self.verbose:
                print(f"arrival_rate={n_veh_finish / np.sum(self.tau) * 100:.2f}%, "
                      f"avg_delay={avg_delay:.2f}, fit={fitness:.2f}")

        return fitness


class TSCProblemActSeq(Problem):
    """Traffic Signal Timing Problem taking action sequence as optimization variable"""
    def __init__(self, scene_name, case, net_input, agent, yellow_dicts, phase_states, verbose=False, simulation_id=None):
        self.scene_name = scene_name
        self.case = case
        self.cfg = config.config_dictionary[self.scene_name]
        self.net_input = np.array(net_input, dtype=np.int)
        self.map_config = map_configs[scene_name]
        self.verbose = verbose
        self.simulation_id = simulation_id
        super().__init__(agent.n_param, xlb=np.zeros(agent.n_param),
                         xub=np.array([sub_act_space.n - 1e-5 for sub_act_space in agent.full_act_space]))
        self.n_ts = agent.n_traffic_signals
        self.act_space = agent.act_space
        self.n_step = agent.n_step
        self.step_length = agent.step_length
        self.n_param = agent.n_param
        self.neighbor_ids = agent.neighbor_ids
        self.non_neighbor_ids = agent.non_neighbor_ids
        self.ts_ids = agent.ts_ids
        self.full_act_space = agent.full_act_space
        self.yellow_dicts = yellow_dicts
        self.phase_states = phase_states
        # prepare net file
        self.net_path = utils.get_net_path(self.scene_name)

        # # for debug
        # print(self.phase_states)
        # print(self.yellow_dicts)

    def prepare(self):
        self.sim_dir = utils.get_sim_cfg_dir_path(scene_name=self.scene_name, case=self.case,
                                                  simulation_id=self.simulation_id, worker_id=None)
        if not os.path.exists(self.sim_dir):
            os.makedirs(self.sim_dir)
        # prepare add file
        self.prepare_add_file()
        # prepare cfg file
        self.prepare_cfg_file()
        # prepare route file
        self.rou_path = utils.get_rou_path(scene_name=self.scene_name, case=self.case, net_input=self.net_input)
        # prepare output file
        self.output_path = os.path.join(self.sim_dir, 'output.xml')

    def prepare_worker(self, worker_id):
        self.sim_dir = utils.get_sim_cfg_dir_path(scene_name=self.scene_name, case=self.case,
                                                  simulation_id=self.simulation_id, worker_id=worker_id)
        if not os.path.exists(self.sim_dir):
            os.makedirs(self.sim_dir)
        # prepare add file
        self.prepare_add_file()
        # prepare cfg file
        self.prepare_cfg_file()
        # prepare route file
        self.rou_path = utils.get_rou_path(scene_name=self.scene_name, case=self.case, net_input=self.net_input)
        # prepare output file
        self.output_path = os.path.join(self.sim_dir, 'output.xml')

    def prepare_add_file(self):
        self.add_path = os.path.join(self.sim_dir, self.scene_name + '.add.xml')
        if not os.path.exists(self.add_path):
            print(f"copying {utils.get_template_add_path(self.scene_name)} to {self.add_path}")
            shutil.copyfile(utils.get_template_add_path(self.scene_name), self.add_path)

    def prepare_cfg_file(self):
        self.cfg_path = os.path.join(self.sim_dir, self.scene_name + '.sumocfg')
        if not os.path.exists(self.cfg_path):
            print(f"copying {utils.get_template_cfg_path()} to {self.cfg_path}")
            shutil.copyfile(utils.get_template_cfg_path(), self.cfg_path)

    def get_xlb(self):
        return self.xlb

    def get_xub(self):
        return self.xub

    def get_dim_sol(self):
        return self.dim

    def sample_x(self):
        return np.random.uniform(self.xlb, self.xub)

    def get_tl_program_code(self, x):
        param = x.reshape(self.n_step, self.n_ts).astype(np.int_)
        # print(param)
        code = "<?xml version=\"1.0\" ?><additional>\n"
        for i, ts_id in enumerate(self.ts_ids):
            code += " " * 4 + f"<tlLogic id=\"{ts_id}\" offset=\"0\" programID=\"1\" type=\"static\">\n"
            compressed_phase_seq = compress_array(np.append(param[:, i].flatten(), np.arange(self.full_act_space[i].n)))
            for j, phase_count in enumerate(compressed_phase_seq):
                state = self.phase_states[ts_id][phase_count[0]]
                duration = (phase_count[1] - 1) * 10 + 7
                code += " " * 8 + f"<phase duration=\"{duration}\" state=\"{state}\"/>\n"
                next_phase_id = compressed_phase_seq[(j + 1) % len(compressed_phase_seq)][0]
                key = str(phase_count[0]) + '_' + str(next_phase_id)

                if key in self.yellow_dicts[ts_id]:
                    # there exists a switch yellow phase
                    yel_idx = self.yellow_dicts[ts_id][key]
                    yel_state = self.phase_states[ts_id][yel_idx]
                    code += " " * 8 + f"<phase duration=\"3\" state=\"{yel_state}\"/>\n"
                else:
                    # does not need a switch phase
                    code += " " * 8 + f"<phase duration=\"3\" state=\"{state}\"/>\n"
            code += " " * 4 + f"</tlLogic>\n"
        code += "</additional>"
        return code
        # print(code)

    def func_with_worker(self, x_workerid):
        x, workerid = x_workerid[0], x_workerid[1]
        x0 = np.array(np.clip(x, self.get_xlb(), self.get_xub()), dtype=np.int)
        # prepare all file paths necessary for simulation
        self.prepare_worker(worker_id=workerid)
        # load route file into '.sumocfg'
        utils.load_cfg_file(self.cfg_path, self.net_path, self.rou_path, self.add_path, self.output_path)
        # load additional file (traffic signal plan) into '.sumocfg'
        utils.load_traffic_light_program_by_text(self.add_path, self.get_tl_program_code(x))
        # run a sumo simulation
        utils.run_simulation_cmd(self.cfg_path)
        # read the results from outputs
        res = utils.read_simulation_output(self.output_path)
        return res[1]


class TSTOProblemFixedCycle(TSTOProblem):
    """Traffic Signal Timing Problem taking duration as optimization variable with fixed cycle"""
    def __init__(self, scene_name, case, net_input, simulation_id=None, cycle_len=90):
        self.min_duration = 5
        self.cycle_len = cycle_len
        super().__init__(scene_name, case, net_input, simulation_id=simulation_id)
        self.dim = self.get_dim_sol()
        self.xlb = self.get_xlb()
        self.xub = self.get_xub()
        # print(self.cfg.n_phases)

    def sample_one(self):
        solution = []
        for n_phase in self.cfg.n_phases:
            residual_cycle_len = self.cycle_len
            for i in range(n_phase - 1):
                solution.append(
                    np.random.randint(self.min_duration, residual_cycle_len - self.min_duration * (n_phase - i - 1) + 1))
                # print(self.min_duration, residual_cycle_len - self.min_duration * (n_phase - i - 1), solution[-1])
                residual_cycle_len -= solution[-1]
            solution.append(residual_cycle_len)
        return np.array(solution).astype(np.int)

    def sample_x(self, nx=1):
        if nx == 1:
            return self.sample_one()
        else:
            return [self.sample_one() for _ in range(nx)]

    def get_dim_sol(self):
        return self.cfg.get_n_duration()

    def get_xlb(self):
        return np.ones(self.get_dim_sol()) * self.min_duration

    def get_xub(self):
        return np.ones(self.get_dim_sol()) * self.cycle_len

    def func_with_worker(self, x_workerid):
        x, workerid = x_workerid[0], x_workerid[1]
        x0 = np.zeros(self.cfg.DIM_DECVAR)
        x0[:len(x)] = x
        x0 = x0.astype(np.int_)
        # prepare all file paths necessary for simulation
        self.prepare_worker(worker_id=workerid)
        # load route file into '.sumocfg'
        utils.load_cfg_file(self.cfg_path, self.net_path, self.rou_path, self.add_path, self.output_path)
        # load additional file (traffic signal plan) into '.sumocfg'
        utils.load_traffic_light_program(x0, self.cfg, self.add_path)
        # run a sumo simulation
        utils.run_simulation_cmd(self.cfg_path)
        # read the results from outputs
        res = utils.read_simulation_output(self.output_path)
        return res[1]


class TrafficPerformanceSurrogateEvaluator(Evaluator):
    def evaluate(self, prob, x, pool=None):
        return prob.func(x)


class TSTOProblemSurrogate(TSTOProblemFixedCycle):
    """Traffic Signal Timing Problem taking duration as optimization variable with fixed cycle"""
    def __init__(self, scene_name, case, net_input, model, simulation_id=None, cycle_len=90):
        super().__init__(scene_name, case, net_input, simulation_id=simulation_id, cycle_len=cycle_len)
        self.model = model

    def func(self, x):
        if x.ndim == 1:
            return self.model.predict([np.append(self.net_input, x)])[0]
        else:
            return self.model.predict(np.c_[np.tile(self.net_input, (x.shape[0], 1)), x])


def compress_array(arr):
    if arr.size == 0:
        return []

    change_points = np.where(arr[:-1] != arr[1:])[0] + 1
    partitions = np.split(arr, change_points)
    compressed = [(part[0], len(part)) for part in partitions]

    if arr.size > 1 and arr[0] == arr[-1]:
        if len(compressed) > 1:
            compressed[0] = (compressed[0][0], compressed[0][1] + compressed[-1][1])
            compressed.pop()

    return compressed


def string_to_worker_id(s):
    hash_number = zlib.crc32(s.encode())
    worker_id = hash_number % (65535 - 8813 + 1)
    return worker_id


def get_agent_and_env(scene_name, case, agent_name, tau, trial=0, n_step=90, step_length=10, param=None, worker_id=None,
                      output_path=None, summary_path=None, agent_path=None):
    map_config = map_configs[scene_name]
    agt_config = agent_configs[agent_name]
    agt_map_config = agt_config.get(scene_name)
    if agt_map_config is not None:
        agt_config = agt_map_config
    alg = agt_config['agent']

    if agent_path is None:
        agent_path = utils.get_saved_agent_path(scene_name, case, alg.__name__, trial)

    if worker_id is None:
        worker_id = trial

    sim_dir = utils.get_agent_simulation_dir(scene_name, case, alg.__name__, worker_id)
    os.makedirs(sim_dir, exist_ok=True)
    sim_cfg_path = utils.get_agent_simulation_cfg_path(scene_name, case, alg.__name__, worker_id)
    log_dir = os.path.join(os.path.dirname(__file__), 'resco_benchmark', 'results' + os.sep)

    if output_path is None:
        output_path = os.path.join(sim_dir, 'output.xml')

    # print("log_dir", log_dir)
    if not os.path.exists(sim_cfg_path):
        shutil.copyfile(os.path.join(config.PROJECT_PATH, "data", "template_rl.sumocfg"), sim_cfg_path)
        utils.load_net_into_cfg(cfgPATH=sim_cfg_path, netPATH=utils.get_net_path(scene_name))
        utils.load_route_file(cfgPATH=sim_cfg_path, rouPATH=utils.get_rou_path(scene_name, case, tau))

    env = MultiSignal(alg.__name__ + '-tr' + str(trial), scene_name, sim_cfg_path, agt_config['state'], agt_config['reward'],
                      route=None, step_length=step_length, yellow_length=3, step_ratio=1, end_time=900,
                      max_distance=agt_config['max_distance'], lights=map_config['lights'], gui=False, log_dir=log_dir,
                      libsumo=False, warmup=0, traffic_inputs=np.atleast_2d(tau), case=case,
                      redirected_output_path=output_path, fixed_timing=None, traffic_signal_timing_dir=sim_dir,
                      worker_id=worker_id, summary_path=summary_path)
    # print(env.observation_space, env.action_space)
    if agent_name in ['IDQN', 'IPPO']:
        f = open(agent_path, 'rb')
        agent = dill.load(f)
        for agt_name in agent.agents.keys():
            agent.agents[agt_name].agent.training = False
    elif agent_name in ['MPLight']:
        f = open(agent_path, 'rb')
        agent = dill.load(f)
        agent.agent.training = False
    elif agent_name in ['MAXWAVE', 'MAXPRESSURE', 'STOCHASTIC']:
        # Get agent id's, observation shapes, and action sizes from env
        agt_config['num_lights'] = len(env.all_ts_ids)
        obs_act = dict()
        for key in env.obs_shape:
            obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
        agent = alg(agt_config, obs_act, scene_name, trial)
    elif agent_name in ['NeuralWave', 'NeuralUrgency', 'NeuralUrgency_v2']:
        # Get agent id's, observation shapes, and action sizes from env
        agt_config['num_lights'] = len(env.all_ts_ids)
        obs_act = dict()
        for key in env.obs_shape:
            obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
        agent = alg(agt_config, obs_act, scene_name, trial, param=param)
    elif agent_name in ['TinyAgent',]:
        agent = alg(env.action_space, n_step=env.end_time // env.step_length, step_length=env.step_length)
        if param is not None:
            agent.load_param(param)
    elif agent_name in ['ActSequence']:
        agent = alg(env.action_space, scene_name, env.all_ts_ids, n_step=n_step, step_length=env.step_length)
        if param is not None:
            agent.load_param(param)
    elif agent_name in ['FixedTiming']:
        agent = alg(scene_name, case, tau)
        fixed_timing = agent.x
        env.fixed_timing = fixed_timing
        print(f"fixed timing is {fixed_timing}")
    else:
        raise NotImplementedError
    return agent, env


if __name__ == "__main__":
    pass
    # t0 = time.time()
    # from multiprocessing import Pool
    # import distribution
    # pool = Pool(20)
    # scene_name = 'net_double'
    # scene_case = 2
    # cfg = config.config_dictionary[scene_name]
    # distribution.prepare(scene_name, scene_case)
    # taus = [distribution.enumerate_train_tau() for i in range(20)]
    # prob_gen = TSTOProblemGenerator(scene_name, scene_case, )
    # prob_List = [prob_gen.get_prob_instance(tau) for tau in taus]
    # prob = prob_gen.get_prob_instance(taus[0])
    # # print('taus', tau)
    # xlb = cfg.get_xlb()
    # xub = cfg.get_xub()
    # X1 = np.random.uniform(xlb, xub, size=(20,xlb.shape[0])).astype(np.int)
    # X2 = np.random.uniform(xlb, xub, size=(20,xlb.shape[0])).astype(np.int)
    # ev1 = TrafficPerformanceEvaluator('spmi', n_workers=20)
    # ev2 = TrafficPerformanceEvaluator('mpmi', n_workers=20)
    # print('fit', ev1.evaluate(prob_List[0], X1, pool=pool))
    # print('fit', ev2.evaluate(prob_List, X1, pool=pool))
    # print('fit', ev1.evaluate(prob_List[0], X2, pool=pool))
    # print('fit', ev2.evaluate(prob_List, X2, pool=pool))
    # print('used time', time.time() - t0)