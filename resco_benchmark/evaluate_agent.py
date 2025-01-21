import utils
import os
import dill
from resco_benchmark.multi_signal import MultiSignal
from resco_benchmark.resco_config.agent_config import agent_configs
from resco_benchmark.resco_config.map_config import map_configs
from resco_benchmark.resco_config.mdp_config import mdp_configs
from resco_benchmark.rewards import wait, queue
import numpy as np
from utils import read_simulation_output
import config
import zlib
import pickle


def string_to_worker_id(s):
    hash_number = zlib.crc32(s.encode())
    worker_id = hash_number % (65535 - 8813 + 1)
    return worker_id


def run(scene_name, case, agent, trial):
    run_agent_on_test_cases(scene_name, case, agent, trial)


def run_agent_on_test_cases(scene_name, case, agent_name, trial):
    train_case = 1
    if train_case == case:
        print('Running on i.i.d. prob. inst.')
    else:
        print(f"Running on o.o.d. prob. inst. from c{train_case} to c{case}")
    use_lmm = False
    full_agent_name = None
    if 'lmm' in agent_name:
        use_lmm = True
        trial = 0
        full_agent_name = agent_name
        agent_name = full_agent_name.split('_')[1]
        # print(agent_name)

    mdp_config = mdp_configs.get(agent_name)
    if mdp_config is not None:
        mdp_map_config = mdp_config.get(scene_name)
        if mdp_map_config is not None:
            mdp_config = mdp_map_config
        mdp_configs[agent_name] = mdp_config

    agt_config = agent_configs[agent_name]
    agt_map_config = agt_config.get(scene_name)
    if agt_map_config is not None:
        agt_config = agt_map_config
    alg = agt_config['agent']

    if mdp_config is not None:
        agt_config['mdp'] = mdp_config
        management = agt_config['mdp'].get('management')
        if management is not None:    # Save some time and precompute the reverse mapping
            supervisors = dict()
            for manager in management:
                workers = management[manager]
                for worker in workers:
                    supervisors[worker] = manager
            mdp_config['supervisors'] = supervisors

    taus = np.load(utils.get_test_tau_path(scene_name, case))
    res = {'delay': [],
           'arrived': [],
           'time_loss': [],
           'queue_length': [],
           'accumulation': [],
           'total_wait': [],
           }

    sim_dir = utils.get_agent_simulation_dir(scene_name, case, alg.__name__, trial)
    os.makedirs(sim_dir, exist_ok=True)
    sim_cfg_path = utils.get_agent_simulation_cfg_path(scene_name, case, alg.__name__, trial)
    output_path = os.path.join(sim_dir, 'output.xml')
    summary_path = os.path.join(sim_dir, 'summary.xml')
    res_path = os.path.join(config.PROJECT_PATH, "data", "test", scene_name, f"c{case}", f"{alg.__name__}-tr{trial}.npy")
    log_dir = os.path.join(os.path.dirname(__file__), 'results' + os.sep)
    # print(output_path, summary_path)
    if use_lmm:
        res_path = os.path.join(config.PROJECT_PATH, "data", "test", scene_name, f"c{case}", f"{full_agent_name}.npy")

    print("log_dir", log_dir)
    if not os.path.exists(sim_cfg_path):
        import shutil
        shutil.copyfile(os.path.join(config.PROJECT_PATH, "data", "template_rl.sumocfg"), sim_cfg_path)
        utils.load_net_into_cfg(cfgPATH=sim_cfg_path, netPATH=utils.get_net_path(scene_name))
        utils.load_route_file(cfgPATH=sim_cfg_path, rouPATH=utils.get_rou_path(scene_name, case, taus[0]))

    connection_str = alg.__name__ + '-tr' + str(trial) + '-' + scene_name + '-c' + str(case) + '-' + \
                     agt_config['state'].__name__ + '-' + agt_config['reward'].__name__

    for i, tau in enumerate(taus):
        print('|---testing problem inst. ' + str(i))
        map_config = map_configs[scene_name]

        env = MultiSignal(alg.__name__ + '-tr' + str(trial),
                          scene_name,
                          sim_cfg_path,
                          agt_config['state'],
                          agt_config['reward'],
                          route=None,
                          step_length=10,
                          yellow_length=3,
                          step_ratio=1,
                          end_time=900,
                          max_distance=agt_config['max_distance'],
                          lights=map_config['lights'],
                          gui=False,
                          log_dir=log_dir,
                          libsumo=False,
                          warmup=0,
                          traffic_inputs=np.array([tau]),
                          case=case,
                          redirected_output_path=output_path,
                          summary_path=summary_path,
                          worker_id=string_to_worker_id(connection_str))

        if i == 0:
            if agent_name in ['IDQN', 'IPPO']:
                agent_path = utils.get_saved_agent_path(scene_name, train_case, alg.__name__, trial)
                f = open(agent_path, 'rb')
                agent = dill.load(f)
                for agt_name in agent.agents.keys():
                    agent.agents[agt_name].agent.training = False
            elif agent_name in ['MPLight']:
                agent_path = utils.get_saved_agent_path(scene_name, train_case, alg.__name__, trial)
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
            elif agent_name in ['FMA2C']:
                agent_model_path = os.path.join(config.PROJECT_PATH, "resco_benchmark", "results",
                                                f"FMA2C-tr{trial}-{scene_name}-c{train_case}"
                                                f"-fma2c-fma2cagent_checkpoint-20000")
                print(agent_model_path)

                agt_config['num_lights'] = len(env.all_ts_ids)
                agt_config['steps'] = False
                # agt_config['model_dir'] = agent_param_dir
                agt_config['model_path'] = agent_model_path
                obs_act = dict()
                for key in env.obs_shape:
                    obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
                agent = alg(agt_config, obs_act, scene_name, trial)
            elif agent_name in ['NeuralUrgency']:
                param = None
                if use_lmm:
                    mdl_path = os.path.join(config.PROJECT_PATH, "data", "mdl", scene_name, f"c{train_case}",
                                            f'{full_agent_name}.pkl')
                    print(f"Using lmm in at {mdl_path}")
                    with open(mdl_path, 'rb') as f:
                        final_mdl = pickle.load(f)
                        data_tau = final_mdl.data_tau
                        data_xopt = final_mdl.data_xopt

                    def knn_map(t, D_t, D_x):
                        return D_x[np.argmin([np.sum(np.abs(t - d_t)) for d_t in D_t])]

                    param = knn_map(tau, data_tau, data_xopt)

                agt_config['num_lights'] = len(env.all_ts_ids)
                obs_act = dict()
                for key in env.obs_shape:
                    obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
                agent = alg(agt_config, obs_act, scene_name, trial, param=param)
            else:
                raise NotImplementedError

        obs = env.reset()
        done = False
        total_rew = 0.
        total_queue_length = []
        total_wait = []
        while not done:
            # print(list(wait(env.signals).values()))
            total_wait.append(-np.sum(list(wait(env.signals).values())))
            total_queue_length.append(-np.sum(list(queue(env.signals).values())))
            act = agent.act(obs)
            obs, rew, done, info = env.step(act)
            total_rew += sum(rew.values())
            # agent.observe(obs, rew, done, info)
        env.close()

        n_arrived, delay = read_simulation_output(outputPATH=output_path, verbose=1)
        time_loss = read_simulation_output(output_path, verbose=0, output_info="timeLoss", average=True)[1]
        res['delay'].append(delay)
        res['arrived'].append(n_arrived / np.sum(tau))
        res['time_loss'].append(time_loss)
        res['accumulation'].append(utils.get_summary_output(summary_path, output_info='accumulation')[::10])
        res['total_wait'].append(total_wait)
        res['queue_length'].append(total_queue_length)

    np.save(res_path, np.array(res['delay']))
    perf_res_path = os.path.join(config.PROJECT_PATH, "data", "perf", scene_name, f"c{case}",
                                 f"{alg.__name__}-tr{trial}.pkl")
    if use_lmm:
        perf_res_path = os.path.join(config.PROJECT_PATH, "data", "perf", scene_name, f"c{case}",
                                     f"{full_agent_name}.pkl")
    os.makedirs(os.path.join(config.PROJECT_PATH, "data", "perf", scene_name, f"c{case}"), exist_ok=True)
    with open(perf_res_path, 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    raise Exception

