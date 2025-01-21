import pathlib
import os
import dill
from resco_benchmark.multi_signal import MultiSignal
from resco_benchmark.resco_config.agent_config import agent_configs
from resco_benchmark.resco_config.map_config import map_configs
from resco_benchmark.resco_config.mdp_config import mdp_configs
import numpy as np
import shutil
import utils
import config
import time
import zlib


def string_to_worker_id(s):
    hash_number = zlib.crc32(s.encode())
    worker_id = hash_number % (65535 - 8813 + 1)
    return worker_id


def run(scene_name, case, agent,  trial, n_episodes=1000, n_train_inst=50):
    t0 = time.time()
    run_trial(scene_name, case, agent, trial, n_episodes=n_episodes, n_train_inst=n_train_inst)
    print(f"Elapsed time={time.time()-t0}")


def run_trial(scene_name, case, agent_name, trial, n_episodes=1000, n_train_inst=50):
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

    map_config = map_configs[scene_name]
    num_steps_eps = int((map_config['end_time'] - map_config['start_time']) / map_config['step_length'])

    # route = map_config['route']
    # if route is not None:
    #     route = os.path.join(args.pwd, route)
    # if args.scene_name == 'grid4x4' or args.scene_name == 'arterial4x4':
    #     if not os.path.exists(route):
    #         raise EnvironmentError("You must decompress environment files defining traffic flow")

    taus = np.load(utils.get_train_tau_path(scene_name, case))
    taus = taus[:n_train_inst]

    sim_dir = utils.get_agent_simulation_dir(scene_name, case, alg.__name__, trial)
    os.makedirs(sim_dir, exist_ok=True)
    sim_cfg_path = utils.get_agent_simulation_cfg_path(scene_name, case, alg.__name__, trial)
    agent_path = utils.get_saved_agent_path(scene_name, case, alg.__name__, trial)
    log_dir = os.path.join(os.path.dirname(__file__), 'results' + os.sep)
    print("log_dir", log_dir)
    if not os.path.exists(sim_cfg_path):
        shutil.copyfile(os.path.join(config.PROJECT_PATH, "data", "template_rl.sumocfg"), sim_cfg_path)
        utils.load_net_into_cfg(cfgPATH=sim_cfg_path, netPATH=utils.get_net_path(scene_name))
        utils.load_route_file(cfgPATH=sim_cfg_path, rouPATH=utils.get_rou_path(scene_name, case, taus[0]))

    connection_str = alg.__name__ + '-tr' + str(trial) + '-' + scene_name + '-c' + str(case) + '-' + \
                     agt_config['state'].__name__ + '-' + agt_config['reward'].__name__

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
                      traffic_inputs=taus,
                      case=case,
                      worker_id=string_to_worker_id(connection_str))

    agt_config['episodes'] = int(n_episodes * 0.8)    # schedulers decay over 80% of steps
    agt_config['steps'] = agt_config['episodes'] * num_steps_eps
    agt_config['log_dir'] = os.path.join(log_dir, env.connection_name)
    agt_config['num_lights'] = len(env.all_ts_ids)

    # Get agent id's, observation shapes, and action sizes from env
    obs_act = dict()
    for key in env.obs_shape:
        obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
    agent = alg(agt_config, obs_act, scene_name, trial)
    total_rews = []
    for _ in range(n_episodes):
        total_rew = 0.
        print('episode', _)
        obs = env.reset()
        done = False
        while not done:
            act = agent.act(obs)
            obs, rew, done, info = env.step(act)
            total_rew += sum(rew.values())
            agent.observe(obs, rew, done, info)
        print('total rew', total_rew)
        total_rews.append(total_rew)
    env.close()

    if agent_name != "FMA2C":
        f = open(agent_path, 'wb')
        dill.dump(agent, f)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Cum_reward')
    plt.title(alg.__name__ + '-tr' + str(trial) + '-' + scene_name)
    plt.plot(np.arange(len(total_rews)),np.array(total_rews))
    plt.savefig(os.path.join("resco_benchmark", "figs",
                             alg.__name__ + '-tr' + str(trial) + '-' + scene_name + '-c' + str(case) + '.png'))
    plt.close()


if __name__ == '__main__':

    pass
