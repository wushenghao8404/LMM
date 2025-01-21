import os,sys,traci
import subprocess
import re
import xml.dom.minidom as minidom
import numpy as np
import config
from datetime import datetime
from copy import deepcopy
import pickle

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def generate_unique_id():
    current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
    unique_id = f"{current_time}{os.getpid()}"
    return unique_id


def get_collected_dataset_path(algo_name, scene_name, case, run_id):
    return os.path.join(config.PROJECT_PATH, "data", "train", scene_name, f"c{case}",
                        algo_name + "_data_r" + str(run_id) + ".npy")


def get_dc_eval_data_path(algo_name, scene_name, case, run_id):
    return os.path.join(config.PROJECT_PATH, "data", "train", scene_name, f"c{case}", "eval",
                        algo_name + "_eval_r" + str(run_id) + ".npy")


def get_train_tau_path(scene_name, case):
    return os.path.join(config.PROJECT_PATH, "data",  "tasks", scene_name, f"train_tau_c{case}.npy")


def get_valid_tau_path(scene_name, case):
    return os.path.join(config.PROJECT_PATH, "data", "tasks", scene_name, f"valid_tau_c{case}.npy")


def get_test_tau_path(scene_name, case):
    return os.path.join(config.PROJECT_PATH, "data", "tasks", scene_name, f"test_tau_c{case}.npy")


def get_rou_path(scene_name, case, net_input):
    return os.path.join(config.PROJECT_PATH, "data", "rou",  scene_name, f"c{case}",
                        get_str_from_flow_input(net_input) + ".rou.xml")


def get_template_add_path(scene_name):
    return os.path.join(config.PROJECT_PATH, "data", "add", scene_name + ".add.xml")


def get_template_cfg_path():
    return os.path.join(config.PROJECT_PATH, "data", "template.sumocfg")


def get_net_path(scene_name):
    return os.path.join(config.PROJECT_PATH, "data", "net", scene_name + ".net.xml")


def get_sim_cfg_dir_path(scene_name, case, simulation_id=None, worker_id=None):
    if simulation_id is None:
        sim_cfg_dir = os.path.join(config.PROJECT_PATH, "tmp", scene_name, f"c{case}")
    else:
        sim_cfg_dir = os.path.join(config.PROJECT_PATH, "tmp", scene_name, f"c{case}", simulation_id)
    if worker_id is None:
        return sim_cfg_dir
    else:
        return os.path.join(sim_cfg_dir, worker_id)


def get_sim_data_dir_path(alg_name, scene_name, case, runid):
    sim_data_dir_path = os.path.join(config.PROJECT_PATH, "data", "sim", scene_name, "c" + str(case),
                                     alg_name, "r" + str(runid))
    return sim_data_dir_path


def get_solution_from_add_file(ADD_PATH):
    dom = minidom.parse(ADD_PATH)
    root = dom.documentElement
    tls = root.getElementsByTagName('tlLogic')
    sol = []
    for tl in tls:
        phases = tl.getElementsByTagName('phase')
        for phase in phases:
            state = phase.getAttribute('state')            
            if 'y' not in state and ('G' in state or 'g' in state):
                sol.append(phase.getAttribute('duration'))

    for tl in tls:
        sol.append(tl.getAttribute('offset'))
    
    return np.array(sol)


def get_agent_simulation_dir(scene_name, case, agent_name, trial):
    return os.path.join(config.PROJECT_PATH, "tmp", scene_name, f"c{case}", "resco_benchmark",
                        f"{agent_name}-tr{trial}-{scene_name}-c{case}")


def get_agent_simulation_cfg_path(scene_name, case, agent_name, trial):
    return os.path.join(config.PROJECT_PATH, "tmp", scene_name, f"c{case}", "resco_benchmark",
                        f"{agent_name}-tr{trial}-{scene_name}-c{case}", f"{scene_name}.sumocfg")


def get_saved_agent_path(scene_name, case, agent_name, trial):
    return os.path.join(config.PROJECT_PATH, "resco_benchmark", "saved_agents",
                        f"{agent_name}-tr{trial}-{scene_name}-c{case}.pkl")


def get_baseline_tsc_hyp(scene_name, case, tsc_name):
    hyp_path = os.path.join(config.PROJECT_PATH, "data", "train", scene_name, f"c{case}", f"{tsc_name.lower()}_hyp.npy")
    if os.path.exists(hyp_path):
        print(f"Using optimized hyperparameter for {tsc_name}")
        return np.load(hyp_path, allow_pickle=True).tolist()['params']
    else:
        print(f"Using default hyperparameter for {tsc_name}")
        return None


def get_str_from_flow_input(netflow_input):
    scene_str = ""
    for i in range(len(netflow_input)):
        scene_str += "_" + str(int(netflow_input[i]))
    return scene_str


def get_mdl_result_path(scene_name, case, res_name):
    return os.path.join(config.PROJECT_PATH, "data", "test", scene_name, f"c{case}", f"{res_name}.npy")


def get_agent_perf_data_path(scene_name, case, agent_name):
    return os.path.join(config.PROJECT_PATH, "data", "perf", scene_name, f"c{case}", f"{agent_name}.pkl")


def get_example_taus(scene_name, case):
    test_taus = np.load(get_test_tau_path(scene_name, case))
    test_taus = test_taus[np.random.RandomState(2024).permutation(test_taus.shape[0])[:10]]
    sorted_ids = np.argsort(np.sum(test_taus, axis=1))
    # sorted_ids = np.argsort(np.sum(test_taus, axis=1))
    # selected_ids = sorted_ids[[int(slice_id) for slice_id in np.linspace(0, len(sorted_ids) - 0.1, 10)]]
    # selected_taus = test_taus[selected_ids]
    return test_taus[sorted_ids]


def get_example_output_path(agent_name, scene_name, case, instance_id, output_type='ouptut'):
    return os.path.join(config.PROJECT_PATH, "data", "out", scene_name, f"c{case}",
                        agent_name, f"{output_type}_i{instance_id}.xml")


def load_network_flow_input(netflow_input, tripPATH):
    dom = minidom.parse(tripPATH)
    root = dom.documentElement
    flows = root.getElementsByTagName('flow')
    for i in range(len(netflow_input)):
        flows[i].setAttribute('number', str(int(netflow_input[i])))

    f = open(tripPATH, "w", encoding="utf-8")
    dom.writexml(f)
    f.close()


def load_durations_into_network_file(durs, cfg, netPATH):
    assert (len(durs) == cfg.NUM_DURATION)
    dom = minidom.parse(netPATH)
    root = dom.documentElement
    tls = root.getElementsByTagName('tlLogic')
    count_duration = 0
    for tl in tls:
        phases = tl.getElementsByTagName('phase')
        for phase in phases:
            state = phase.getAttribute('state')
            if 'y' not in state and ('G' in state or 'g' in state):
                phase.setAttribute('duration', str(durs[count_duration]))
                count_duration += 1
            else:
                phase.setAttribute('duration', '3')

    if count_duration != cfg.NUM_DURATION:
        raise Exception("Unmatched duration number")

    f = open(netPATH, "w", encoding="utf-8")
    dom.writexml(f)
    f.close()


def load_route_file(cfgPATH, rouPATH):
    dom = minidom.parse(cfgPATH)
    root = dom.documentElement
    input = root.getElementsByTagName('input')[0]
    input_route_file = input.getElementsByTagName('route-files')[0]
    input_route_file.setAttribute('value',rouPATH)
    f = open(cfgPATH, "w", encoding="utf-8")
    dom.writexml(f)
    f.close()


def load_net_into_cfg(cfgPATH, netPATH):
    dom = minidom.parse(cfgPATH)
    root = dom.documentElement
    input = root.getElementsByTagName('input')[0]
    input_net_file = input.getElementsByTagName('net-file')[0]
    input_net_file.setAttribute('value', netPATH)
    f = open(cfgPATH, "w", encoding="utf-8")
    dom.writexml(f)
    f.close()


def load_cfg_file(cfgPATH, netPATH, rouPATH, addPATH, outPATH):
    dom = minidom.parse(cfgPATH)
    root = dom.documentElement
    input = root.getElementsByTagName('input')[0]
    input_net_file = input.getElementsByTagName('net-file')[0]
    input_net_file.setAttribute('value', netPATH)
    input_route_file = input.getElementsByTagName('route-files')[0]
    input_route_file.setAttribute('value',rouPATH)
    input_add_file = input.getElementsByTagName('additional-files')[0]
    input_add_file.setAttribute('value', addPATH)
    output = root.getElementsByTagName('output')[0]
    output_file = output.getElementsByTagName('tripinfo-output')[0]
    output_file.setAttribute('value', outPATH)
    f = open(cfgPATH, "w", encoding="utf-8")
    dom.writexml(f)
    f.close()


def load_traffic_light_program(sol, cfg, addPATH):
    assert(len(sol) == cfg.DIM_DECVAR)
    dom = minidom.parse(addPATH)
    root = dom.documentElement
    tls = root.getElementsByTagName('tlLogic')
    count_duration = 0
    count_offset = 0
    for tl in tls:
        phases = tl.getElementsByTagName('phase')
        for phase in phases:
            state = phase.getAttribute('state')            
            if 'y' not in state and ('G' in state or 'g' in state):
                phase.setAttribute('duration', str(sol[count_duration]))
                count_duration += 1
            else:
                phase.setAttribute('duration', '3')
                
    if len(tls) > 1:
        for tl in tls:
            tl.setAttribute('offset', str(sol[count_duration + count_offset]))
            count_offset += 1

    if count_duration != cfg.NUM_DURATION:
        raise Exception("Unmatched duration number")
    if count_offset != cfg.NUM_OFFSET:
        raise Exception("Unmatched offset number")

    f = open(addPATH, "w", encoding="utf-8")
    dom.writexml(f)
    f.close()


def load_traffic_light_program_by_Webster(addPATH, offsetPATH, tlsPATH):
    dom = minidom.parse(addPATH)
    root = dom.documentElement
    tls = root.getElementsByTagName('tlLogic')
    
    dom1 = minidom.parse(offsetPATH)
    root1 = dom1.documentElement
    tls1 = root1.getElementsByTagName('tlLogic')
    
    dom2 = minidom.parse(tlsPATH)
    root2 = dom2.documentElement
    tls2 = root2.getElementsByTagName('tlLogic')
    
    for tl, tl1 in tls, tls1:
        phases = tl.getElementsByTagName('phase')
        phases1 = tl1.getElementsByTagName('phase')
        for phase, phase1 in phases, phases1:
            phase.setAttribute('duration', phase1.getAttribute('duration'))
    
    for tl, tl2 in tls, tls2:
        tl.setAttribute('offset', tl2.getAttribute('offset'))

    with open(addPATH, "w", encoding="utf-8") as f:
        dom.writexml(f)
        f.close()


def load_traffic_light_program_by_text(addPATH, text):
    with open(addPATH, "w", encoding="utf-8") as f:
        f.write(text)


def load_dataset(coll_alg_name, scene_name, scene_case, run_id=0):
    cfg = config.config_dictionary[scene_name]
    dataset_path = os.path.join(config.PROJECT_PATH, "data", "train", scene_name, "c" + str(scene_case),
                                     coll_alg_name + "_data_r" + str(run_id) + ".npy")
    dataset = np.load(dataset_path, allow_pickle=True)
    return dataset[:, :cfg.DIM_PROB_FEATURE], dataset[:, cfg.DIM_PROB_FEATURE:]


def load_simdata(coll_alg_name, scene_name, scene_case, runid, net_input):
    datapath = get_sim_data_dir_path(coll_alg_name, scene_name, scene_case, runid) + coll_alg_name + \
               get_str_from_flow_input(net_input) + '.npy'
    simdata = np.load(datapath, allow_pickle=True)
    return simdata[:, :-1], simdata[:, -1]


def load_optim_based_map_model_data(mdl_name, scene_name, case, n_run=10):
    if mdl_name == 'lmm20':
        n_run = 1
    test_taus = np.load(get_test_tau_path(scene_name, case))
    n_tau = test_taus.shape[0]
    data_dir = os.path.join(config.PROJECT_PATH, "data", "test", scene_name, f"c{case}", mdl_name)
    data = []
    for i in range(n_tau):
        data_on_tau = []
        for r in range(n_run):
            data_on_tau.append(np.load(os.path.join(data_dir, f"i{i}_r{r}.npy")))
        data.append(data_on_tau)
    return data


def load_optim_based_map_model_run_info(mdl_name, scene_name, case):
    data_dir = os.path.join(config.PROJECT_PATH, "data", "test", scene_name, f"c{case}", mdl_name)
    info = np.load(os.path.join(data_dir, f"info.npy"), allow_pickle=True).tolist()
    return info


def load_agent_perf_data(scene_name, case, agent_name, metric):
    with open(get_agent_perf_data_path(scene_name, case, agent_name), 'rb') as f:
        data = pickle.load(f)
    return data[metric]


def run_simulation_traci(cfgPATH):
    # Frequently causing error: Error: tcpip::Socket::accept() Unable to create listening socket: Address already in use
    sumoCmd = ["sumo", "-c", cfgPATH]
    traci.start(sumoCmd)
    step = 0
    while step < 900:
        # print("step",step)
        traci.simulationStep()
        step += 1
    traci.close(sumoCmd)
    
    
def run_simulation_cmd(cfgPATH):
    subprocess.call(["sumo", "-c", cfgPATH, "--no-step-log", "--no-warnings"]) 
    

def read_simulation_output(outputPATH, verbose=0, output_info="waitingTime", average=True):
    f_output = open(outputPATH)
    st = f_output.read()

    if output_info == "arrived":
        reg = "<tripinfo" + "\\s.*?" + "waitingTime" + "=[\"]?(.*?)[\"]?\\s.*?>"
    else:
        reg = "<tripinfo" + "\\s.*?" + output_info + "=[\"]?(.*?)[\"]?\\s.*?>"

    compile_name = re.compile(reg, re.M)
    res_name = compile_name.findall(st)
    f_output.close()

    vals = []
    for val in res_name:
        vals.append(float(val))
    n = len(vals)

    if n == 0:
        n, res = 1, 900
        return n, res
        # raise Exception("Something wrong must have happened, causing no cars finish trips")
    elif average:
        res = np.mean(vals)
        if verbose == 1:
            print(f"mean {output_info} = ", res, "s of ", n, " vehicles")
        if output_info == "arrived":
            return res, n
        else:
            return n, res
    else:
        return vals


def read_simulation_summary(summaryPATH, output_info="running"):
    with open(summaryPATH) as f_summary:
        st = f_summary.read()
    reg = "<step" + "\\s.*?" + re.escape(output_info) + "=[\"]?(.*?)[\"]?\\s.*?>"
    compile_name = re.compile(reg, re.M)
    res_name = compile_name.findall(st)
    vals = []

    for val in res_name:
        vals.append(float(val))

    n = len(vals)
    # print(output_info, vals, n)
    if n == 0:
        raise Exception("Something wrong must have happened")
    else:
        return vals


def get_summary_output(summaryPATH, output_info="accumulation"):
    if output_info == "accumulation":
        n_runnings = read_simulation_summary(summaryPATH, output_info="running")
        n_haltings = read_simulation_summary(summaryPATH, output_info="halting")
        return np.array(n_runnings) + np.array(n_haltings)
    elif output_info == "arrived":
        return np.array(read_simulation_summary(summaryPATH, output_info="arrived"))
    else:
        raise NotImplementedError


def merge_dataset(scene_name, case_list, alg_name, runid=0):
    cfg = config.config_dictionary[scene_name]
    data_tau = None
    data_xopt = None
    for case in case_list:
        if alg_name in ['webster']:
            data_path = cfg.SCENE_PATH + 'train_data/' + alg_name + '_data_c' + str(case) + '.npy'
        else:
            data_path = cfg.SCENE_PATH + 'train_data/' + alg_name + '_data_c' + str(case) + '_r' + str(runid) + '.npy'
        dataset = np.load(data_path, allow_pickle=True)
        if data_tau is None:
            data_tau = dataset[:, :cfg.DIM_PROB_FEATURE]
            data_xopt = dataset[:, cfg.DIM_PROB_FEATURE:]
        else:
            data_tau = np.r_[data_tau, dataset[:,:cfg.DIM_PROB_FEATURE]]
            data_xopt = np.r_[data_xopt, dataset[:,cfg.DIM_PROB_FEATURE:]]
    return data_tau, data_xopt


def check_error_log_file(directory):
    print(f"Checking directory '{directory}':")
    error_files = []

    for filename in os.listdir(directory):
        if filename.endswith('.err'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r') as file:
                    content = file.read().lower()
                    if 'error' in content and 'traceback' in content:
                        error_files.append(filename)
            except Exception as e:
                print(f"Can read {filename}: {e}")

    if error_files:
        print(f"The files are reporting 'error':")
        for error_file in error_files:
            print(error_file)
    else:
        print("There are no files reporting error.")


def rad_based_clustering(X, rad):
    clusts =[]
    labels = []
    for i, x in enumerate(X):
        if len(labels) == 0:
            labels.append(0)
            clusts.append(x)
        else:
            found = False
            for i in range(len(clusts)):
                if np.sum(np.abs(x - clusts[i])) < rad:
                    found = True
                    labels.append(i)
                    break
            if not found:
                labels.append(len(clusts))
                clusts.append(x)
    return np.array(labels)


def remove_duplicate(X, Y=None, eps=1.0):
    retX, retY = [], []
    for i, x in enumerate(X):
        if len(retX) == 0:
            retX.append(x)
            if Y is not None:
                retY.append(Y[i])
        else:
            found = False
            for rx in retX:
                if np.sum(np.abs(x - rx)) < eps:
                    found = True
                    break
            if not found:
                retX.append(x)
                if Y is not None:
                    retY.append(Y[i])
    if Y is not None:
        return np.array(retX), np.array(retY)
    else:
        return np.array(retX)


def remove_and_count_duplicate(x,labels,eps=0.1):
    nx = []
    nc = np.max(labels) + 1
    counts = []
    for i in range(x.shape[0]):
        if len(nx) == 0:
            nx.append(x[i,:])
            subcount = np.zeros(nc)
            subcount[labels[i]] += 1
            counts.append(subcount)
        else:
            flag = False
            for j in range(len(nx)):
                if np.sqrt(np.sum((x[i, :] - nx[j]) ** 2)) < eps:
                    subcount = counts[j]
                    subcount[labels[i]] += 1
                    counts[j] = deepcopy(subcount)
                    flag = True
                    break
            if not flag:
                nx.append(x[i, :])
                subcount = np.zeros(nc)
                subcount[labels[i]] += 1
                counts.append(subcount)
    nlabels = np.array([np.argmax(subcount) for subcount in counts])
    ncounts = np.array([np.sum(subcount) for subcount in counts])
    nx = np.array(nx)
    return nx, ncounts, nlabels


if __name__ == "__main__":
    '''
        Synthetic Scenes:
        p1: net_single1
        p2: net_single2
        p3: net_double
        p4: net_2x2grid
        p5: net_3x3grid

        Real-world Scenes:
        p1: cologne1
        p2: cologne3
        p3: cologne8
        p4: ingolstadt1
        p5: ingolstadt7
        p6: ingolstadt21
    '''
    raise Exception('Do not run this file directly')
    # remove_net_data()
    # load_durations_into_net([20,30],config.config_dictionary['net_single1'],config.config_dictionary['net_single1'].SCENE_PATH + 'net_single1.net.xml')
