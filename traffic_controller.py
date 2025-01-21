import utils
import xml.dom.minidom as minidom
import subprocess
import numpy as np
import config
import os
from baseline_tsc.networkdata import NetworkData
from baseline_tsc.sumosim import SumoSim
from multiprocessing import Pool
TLS_CA_PATH = config.SUMO_PATH + "tools/tlsCycleAdaptation.py"
TLS_CO_PATH = config.SUMO_PATH + "tools/tlsCoordinator.py"
PROJECT_PATH = config.PROJECT_PATH


def SumoWebster(scene_name, case, net_input, verbose=0, worker_id=0):
    cfg = config.config_dictionary[scene_name]
    net_dir = os.path.join(config.PROJECT_PATH, "tmp", scene_name, f"c{case}", "sumowebster", f"{worker_id}")
    os.makedirs(net_dir, exist_ok=True)
    NET_PATH = os.path.join(net_dir, f"{scene_name}.net.xml")
    if not os.path.exists(NET_PATH):
        import shutil
        print(f"copying {utils.get_net_path(scene_name)} to {NET_PATH}")
        shutil.copyfile(utils.get_net_path(scene_name), NET_PATH)
    ROU_FILE_PATH = utils.get_rou_path(scene_name, case, net_input)
    # optimize best cycle length and durations by Webster equation
    TLS_CYCLE_PATH = os.path.join(net_dir, f"tlscycle.add.xml")
    subprocess.call(["python", TLS_CA_PATH,
                     "-n", NET_PATH,
                     "-r", ROU_FILE_PATH,
                     "-o", TLS_CYCLE_PATH,
                     "-g", "5",
                     ])
    sol = []
    # read cycle durations
    dom = minidom.parse(TLS_CYCLE_PATH)
    root = dom.documentElement
    tls = root.getElementsByTagName('tlLogic')
    for tl in tls:
        phases = tl.getElementsByTagName('phase')
        for phase in phases:
            state = phase.getAttribute('state')
            if 'y' not in state and ('G' in state or 'g' in state):
                sol.append(int(float(phase.getAttribute('duration'))))
    utils.load_durations_into_network_file(sol, cfg, NET_PATH)

    # optimize offsets by green wave
    TLS_OFFSET_PATH = os.path.join(net_dir, "tlsoffset.add.xml")
    subprocess.call(["python", TLS_CO_PATH,
                     "-n", NET_PATH,
                     "-r", ROU_FILE_PATH,
                     "-o", TLS_OFFSET_PATH,
                     ])
    # read offset
    dom = minidom.parse(TLS_OFFSET_PATH)
    root = dom.documentElement
    tls = root.getElementsByTagName('tlLogic')
    if cfg.NUM_OFFSET > 1:
        for tl in tls:
            sol.append(int(float(tl.getAttribute('offset'))))
    xlb = cfg.get_xlb()
    xub = cfg.get_xub()
    if verbose == 1:
        print("net input:", np.array(net_input))
        print("solution:", np.array(sol))
    return np.clip(np.array(sol), xlb, xub)


def SumoGreenWave(scene_name, case, net_input, cycle_length, verbose=0, worker_id=0):
    cfg = config.config_dictionary[scene_name]
    net_dir = os.path.join(config.PROJECT_PATH, "tmp", scene_name, f"c{case}", "sumogreenwave", f"{worker_id}")
    os.makedirs(net_dir, exist_ok=True)
    NET_PATH = os.path.join(net_dir, f"{scene_name}.net.xml")
    if not os.path.exists(NET_PATH):
        import shutil
        print(f"copying {utils.get_net_path(scene_name)} to {NET_PATH}")
        shutil.copyfile(utils.get_net_path(scene_name), NET_PATH)
    ROU_FILE_PATH = utils.get_rou_path(scene_name, case, net_input)
    TLS_OFFSET_PATH = os.path.join(net_dir, "tlsoffset.add.xml")
    dom = minidom.parse(NET_PATH)
    root = dom.documentElement
    tls = root.getElementsByTagName('tlLogic')
    sol = []
    lost_time = 4
    subprocess.call(["python", TLS_CO_PATH,
                     "-n", NET_PATH,
                     "-r", ROU_FILE_PATH,
                     "-o", TLS_OFFSET_PATH,
                     ])

    # read cycle durations
    for tl in tls:
        phases = tl.getElementsByTagName('phase')
        num_effective_state = 0
        for phase in phases:
            state = phase.getAttribute('state')            
            if 'y' not in state and ('G' in state or 'g' in state):
                num_effective_state += 1
        
        for i in range(num_effective_state):
            sol.append(int((cycle_length - lost_time * num_effective_state) / num_effective_state))
    utils.load_durations_into_network_file(sol, cfg, NET_PATH)

    # read offset
    if cfg.NUM_OFFSET > 1:
        dom = minidom.parse(TLS_OFFSET_PATH)
        root = dom.documentElement
        tls = root.getElementsByTagName('tlLogic')
        for tl in tls:
            sol.append(int(float(tl.getAttribute('offset'))))
    sol = np.array(sol)
    xlb = cfg.get_xlb()
    xub = cfg.get_xub()
    if verbose == 1:
        print("net input:", np.array(net_input))
        print("solution:", np.array(sol))
    return np.clip(np.array(sol), xlb, xub)


def wrapped_SumoWebster(input_args):
    return SumoWebster(scene_name=input_args['scene_name'], case=input_args['case'], net_input=input_args['net_input'],
                       verbose=input_args['verbose'], worker_id=input_args['worker_id'])


def wrapped_SumoGreenWave(input_args):
    return SumoGreenWave(scene_name=input_args['scene_name'], case=input_args['case'],
                         net_input=input_args['net_input'], cycle_length=input_args['cycle_length'],
                         verbose=input_args['verbose'], worker_id=input_args['worker_id'])


def get_Webster_delay(scene_name, case, net_input, worker_id, params=None):
    # re-implementation of https://github.com/docwza/sumolights
    NET_PATH = utils.get_net_path(scene_name)
    ROU_FILE_PATH = utils.get_rou_path(scene_name, case, net_input)
    worker_dir = os.path.join(config.PROJECT_PATH, "tmp", scene_name, f"c{case}", "webster", f"{worker_id}")
    os.makedirs(worker_dir, exist_ok=True)
    CFG_PATH = os.path.join(worker_dir, f"{scene_name}.sumocfg")
    OUTPUT_PATH = os.path.join(worker_dir, f"output.xml")

    if not os.path.exists(CFG_PATH):
        import shutil
        print(f"copying {utils.get_template_cfg_path()} to {CFG_PATH}")
        shutil.copyfile(utils.get_template_cfg_path(), CFG_PATH)

    utils.load_net_into_cfg(CFG_PATH, NET_PATH)
    utils.load_route_file(CFG_PATH, ROU_FILE_PATH)
    netdata = NetworkData(NET_PATH).get_net_data()
    update_freq = config.default_baseline_tsc_hp['webster']['update_freq']

    if params is not None:
        update_freq = params['update_freq']

    args = {'port'       : 9000,
            'r'          : 3,
            'y'          : 2,
            'g_min'      : 5,
            'c_min'      : 60,
            'c_max'      : 180,
            'sat_flow'   : 0.38,
            'update_freq': update_freq,
            'output_path': OUTPUT_PATH,}
    sim = SumoSim(CFG_PATH, 900, 'webster', True, netdata, args, worker_id,
                  connection_name=f"webster_{scene_name}_c{case}")
    sim.gen_sim()
    sim.create_tsc()
    sim.run()
    sim.close()
    # read the results from outputs
    res = utils.read_simulation_output(OUTPUT_PATH)
    # print('avg delay of Webster is', res[1])
    return res[1]


def get_Maxpressure_delay(scene_name, case, net_input, worker_id, params=None):
    # re-implementation of https://github.com/docwza/sumolights
    NET_PATH = utils.get_net_path(scene_name)
    ROU_FILE_PATH = utils.get_rou_path(scene_name, case, net_input)
    worker_dir = os.path.join(config.PROJECT_PATH, "tmp", scene_name, f"c{case}", "maxpressure", f"{worker_id}")
    os.makedirs(worker_dir, exist_ok=True)
    CFG_PATH = os.path.join(worker_dir, f"{scene_name}.sumocfg")
    OUTPUT_PATH = os.path.join(worker_dir, f"output.xml")

    if not os.path.exists(CFG_PATH):
        import shutil
        print(f"copying {utils.get_template_cfg_path()} to {CFG_PATH}")
        shutil.copyfile(utils.get_template_cfg_path(), CFG_PATH)

    utils.load_net_into_cfg(CFG_PATH, NET_PATH)
    utils.load_route_file(CFG_PATH, ROU_FILE_PATH)
    netdata = NetworkData(NET_PATH).get_net_data()
    args = {'port' : 9000,
            'r'    : 3,
            'y'    : 2,
            'g_min': 5,
            'output_path': OUTPUT_PATH,}
    sim = SumoSim(CFG_PATH, 900, 'maxpressure', True, netdata, args, worker_id,
                  connection_name=f"maxpressure_{scene_name}_c{case}")
    sim.gen_sim()
    sim.create_tsc()
    sim.run()
    sim.close()
    # read the results from outputs
    res = utils.read_simulation_output(OUTPUT_PATH)
    # print('avg delay of Maxpressure is', res[1])
    return res[1]


def get_SOTL_delay(scene_name, case, net_input, worker_id, params=None):
    # re-implementation of https://github.com/docwza/sumolights
    NET_PATH = utils.get_net_path(scene_name)
    ROU_FILE_PATH = utils.get_rou_path(scene_name, case, net_input)
    worker_dir = os.path.join(config.PROJECT_PATH, "tmp", scene_name, f"c{case}", "sotl", f"{worker_id}")
    os.makedirs(worker_dir, exist_ok=True)
    CFG_PATH = os.path.join(worker_dir, f"{scene_name}.sumocfg")
    OUTPUT_PATH = os.path.join(worker_dir, f"output.xml")

    if not os.path.exists(CFG_PATH):
        import shutil
        print(f"copying {utils.get_template_cfg_path()} to {CFG_PATH}")
        shutil.copyfile(utils.get_template_cfg_path(), CFG_PATH)

    utils.load_net_into_cfg(CFG_PATH, NET_PATH)
    utils.load_route_file(CFG_PATH, ROU_FILE_PATH)
    netdata = NetworkData(NET_PATH).get_net_data()

    theta = config.default_baseline_tsc_hp['sotl']['theta']
    omega = config.default_baseline_tsc_hp['sotl']['omega']
    mu = config.default_baseline_tsc_hp['sotl']['mu']

    if params is not None:
        theta = params['theta']
        omega = params['omega']
        mu = params['mu']

    args = {'port' : 9000,
            'r'    : 3,
            'y'    : 2,
            'g_min': 5,
            'theta': theta,
            'omega': omega,
            'mu'   : mu,
            'output_path': OUTPUT_PATH,}
    sim = SumoSim(CFG_PATH, 900, 'sotl', True, netdata, args, worker_id,
                  connection_name=f"sotl_{scene_name}_c{case}")
    sim.gen_sim()
    sim.create_tsc()
    sim.run()
    sim.close()
    # read the results from outputs
    res = utils.read_simulation_output(OUTPUT_PATH)
    # print('avg delay of SOTL is', res[1])
    return res[1]


def wrapped_Webster(args):
    return get_Webster_delay(args['scene_name'], args['case'], args['tau'], args['worker_id'], params=args['param'])


def wrapped_Maxressure(args):
    return get_Maxpressure_delay(args['scene_name'], args['case'], args['tau'], args['worker_id'], params=args['param'])


def wrapped_SOTL(args):
    return get_SOTL_delay(args['scene_name'], args['case'], args['tau'], args['worker_id'], params=args['param'])


def get_baseline_tsc_delays_with_multiproc(planner_name, scene_name, case, taus, n_workers=20, pool=None, hyperparam=None):
    if pool is None:
        pool = Pool(n_workers)
    if planner_name == 'Webster':
        tsc_func = wrapped_Webster
    elif planner_name == 'Maxpressure':
        tsc_func = wrapped_Maxressure
    elif planner_name == 'SOTL':
        tsc_func = wrapped_SOTL
    else:
        raise Exception('Unexpected baseline planner name')
    n_tau = taus.shape[0]
    delays = []
    for i in range(n_tau // n_workers + 1):
        input_params = [{'scene_name': scene_name,
                         'case': case,
                         'tau': tau,
                         'worker_id': wid,
                         'param': hyperparam,
                         } for wid, tau in enumerate(taus[i * n_workers:(i + 1) * n_workers, :])]
        delays += pool.map(tsc_func, input_params)
    delays = np.array(delays)
    return delays


def collect_sumo_tsc_with_multiproc(planner_name, scene_name, case, taus, pool, n_workers=20, params=None):
    print('---------', f'generating {planner_name} plan', '---------')
    if planner_name == 'sumowebster':
        wrapped_tsc = wrapped_SumoWebster
    elif planner_name == 'sumogreenwave':
        wrapped_tsc = wrapped_SumoGreenWave
    else:
        raise Exception('Unexpected baseline planner name')
    n_tau = taus.shape[0]
    sols = []
    for i in range(n_tau // n_workers + 1):
        batch_taus = taus[i * n_workers:(i + 1) * n_workers, :]
        input_args_List = []
        for wid, tau in enumerate(batch_taus):
            input_args = {'scene_name': scene_name,
                          'case': case,
                          'net_input': tau,
                          'verbose': 0,
                          'worker_id': str(wid)}
            if planner_name == 'sumogreenwave':
                input_args['cycle_length'] = params['cycle_length']
            input_args_List.append(input_args)
        sols += pool.map(wrapped_tsc, input_args_List)
    sols = np.array(sols)

    tsc_data_dir = os.path.join(config.PROJECT_PATH, "data", "test", scene_name, f"c{case}", "sumo_tsc")
    os.makedirs(tsc_data_dir, exist_ok=True)
    if planner_name == 'sumowebster':
        data_file_name = f"{planner_name}_data.npy"
    elif planner_name == 'sumogreenwave':
        cycle_length = params['cycle_length']
        data_file_name = f"{planner_name}_c{cycle_length}_data.npy"
    else:
        raise Exception('Unexpected baseline planner name')
    np.save(os.path.join(tsc_data_dir, data_file_name), np.c_[taus, sols])


def collect_sumo_tsc_plans_on_test_cases(planner_name, scene_name, case, n_workers=20, pool=None, hyperparam=None):
    if pool is None:
        pool = Pool(n_workers)
    taus = np.load(utils.get_test_tau_path(scene_name, case))
    collect_sumo_tsc_with_multiproc(planner_name, scene_name, case, taus, pool, n_workers=20, params=hyperparam)
    # pool.close()


def read_sumo_tsc_plan(planner_name, scene_name, case, net_input, params=None):
    if planner_name == 'sumowebster':
        data_file_name = f"{planner_name}_data.npy"
    elif planner_name == 'sumogreenwave':
        cycle_length = params['cycle_length']
        data_file_name = f"{planner_name}_c{cycle_length}_data.npy"
    else:
        raise Exception('Unexpected baseline planner name')
    cfg = config.config_dictionary[scene_name]
    dataset = np.load(os.path.join(config.PROJECT_PATH, "data", "test", scene_name, f"c{case}", "sumo_tsc", data_file_name))
    tdists = np.array([np.sum(np.abs(tau - net_input)) for tau in dataset[:, :cfg.DIM_PROB_FEATURE]])
    index = np.argmin(tdists)
    if np.min(tdists) >= 1.0:
        raise Exception("No match network input in Webster's method generated data.")
    return dataset[index, cfg.DIM_PROB_FEATURE:]


if __name__ == "__main__":
    '''
      p1: net_single1
      p2: net_single2
      p3: net_double
      p4: net_2x2grid
      p5: net_3x3grid
    '''
    # batch_SumoWebster('test')
    pass
    raise Exception('Do not run this file')
