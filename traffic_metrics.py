import numpy as np
import config
import traffic_problem
import map_model
import time
import utils
import os


def model_delay_on_validation_cases(scene_name, case, mdl_map, n_reps=1, n_workers=1, pool=None, simulation_id=None):
    if pool is None:
        from multiprocessing import Pool
        pool = Pool(20)
    pg = traffic_problem.TSTOProblemGenerator(scene_name, case, simulation_id=simulation_id)
    taus = np.load(utils.get_valid_tau_path(scene_name, case))
    avg_delays = np.zeros(taus.shape[0])
    for i in range(n_reps):
        X = np.array([mdl_map(tau) for tau in taus])
        prob_List = [pg.get_prob_instance(tau) for tau in taus]
        ev = traffic_problem.TrafficPerformanceEvaluator('mpmi', n_workers=n_workers)
        delays = np.array(ev.evaluate(prob_List, X, pool=pool))
        # for i in range(len(delays)):
        #     print(i,taus[i,:],delays[i])
        avg_delays += delays
    avg_delays = avg_delays / n_reps
    return avg_delays


def model_delay_on_test_cases(scene_name, case, mdl_map, n_reps=1, n_workers=1, pool=None, return_info=False,
                              simulation_id=None):
    if pool is None:
        from multiprocessing import Pool
        pool = Pool(20)
    pg = traffic_problem.TSTOProblemGenerator(scene_name, case, simulation_id=simulation_id)
    taus = np.load(utils.get_test_tau_path(scene_name, case))
    avg_delays = np.zeros(taus.shape[0])
    info = {'wall_clock_time': 0.0}
    for i in range(n_reps):
        t0 = time.time()
        X = np.array([mdl_map(tau) for tau in taus])
        t1 = time.time()
        info['wall_clock_time'] += (t1 - t0)
        prob_List = [pg.get_prob_instance(tau) for tau in taus]
        ev = traffic_problem.TrafficPerformanceEvaluator('mpmi', n_workers=n_workers)
        delays = np.array(ev.evaluate(prob_List, X, pool=pool))
        avg_delays += delays
    avg_delays = avg_delays / n_reps
    info['wall_clock_time'] /= (n_reps * taus.shape[0])
    if return_info:
        return avg_delays, info
    else:
        return avg_delays


def get_optimizer_delays_on_test_cases(scene_name, case, optimizer_name,
                                       n_workers=20, n_reps=1, return_info=False, pool=None, simulation_id=None):
    if pool is None:
        from multiprocessing import Pool
        pool = Pool(20)
    taus = np.load(utils.get_test_tau_path(scene_name, case))
    avg_delays = np.zeros(taus.shape[0])
    info = {'wall_clock_time': 0.0}
    for i, tau in enumerate(taus):
        # optimize the problem
        for run_id in range(n_reps):
            print("|", ("runid " + str(run_id)).ljust(config.display_bar_len), "|")
            opt_algo = map_model.get_optimization_map(optimizer_name)(scene_name, case, i,
                                                                      n_workers=n_workers,
                                                                      run_id=run_id,
                                                                      save=True,
                                                                      maxnfe=config.search_fes[scene_name],
                                                                      simulation_id=simulation_id)
            t0 = time.time()
            opt_algo.map(tau, pool)
            t1 = time.time()
            info['wall_clock_time'] += (t1 - t0)
            algo_data = np.load(opt_algo.algo_data_path)
            avg_delays[i] += np.min(algo_data[:, -1])
            print("|", ("delay=" + str(np.min(algo_data[:, -1]).round(2))).ljust(config.display_bar_len), "|")
    avg_delays = avg_delays / n_reps
    info['wall_clock_time'] /= (n_reps * taus.shape[0])
    if return_info:
        return avg_delays, info
    else:
        return avg_delays


def get_optimizer_delays_from_data(scene_name, case, opt_algo_name, n_reps=10, maxnfe=-1):
    taus = np.load(utils.get_test_tau_path(scene_name, case))
    avg_delays = np.zeros(taus.shape[0])
    for inst_id, tau in enumerate(taus):
        for run_id in range(n_reps):
            algo_data_dir = os.path.join(config.PROJECT_PATH, "data", "test", scene_name, f"c{case}", opt_algo_name)
            algo_data_path = os.path.join(algo_data_dir, 'i' + str(inst_id) + '_r' + str(run_id) + '.npy')
            try:
                algo_data = np.load(algo_data_path)
            except:
                return None
            avg_delays[inst_id] += np.min(algo_data[:maxnfe, -1])
    avg_delays = avg_delays / n_reps
    return avg_delays


def get_baseline_tsc_delay_on_test_cases(map_model_name, scene_name, case, n_workers=20, pool=None):
    from traffic_controller import get_baseline_tsc_delays_with_multiproc
    if map_model_name == 'baseline_webster':
        planner_name = 'Webster'
    elif map_model_name == 'baseline_maxpressure':
        planner_name = 'Maxpressure'
    elif map_model_name == 'baseline_sotl':
        planner_name = 'SOTL'
    else:
        raise Exception('Unexpected baseline planner name')
    taus = np.load(utils.get_test_tau_path(scene_name, case))
    delays = get_baseline_tsc_delays_with_multiproc(planner_name, scene_name, case, taus,
                                                    n_workers=n_workers,
                                                    pool=pool,
                                                    hyperparam=utils.get_baseline_tsc_hyp(scene_name, case, planner_name))
    return delays


def get_sumo_tsc_delay_on_test_cases(map_mdl_name, scene_name, case, n_workers=20, pool=None, simulation_id=None):
    from traffic_controller import read_sumo_tsc_plan
    if map_mdl_name == 'sumowebster':
        mdl_map = lambda x: read_sumo_tsc_plan("sumowebster", scene_name, case, x)
    elif map_mdl_name == 'sumogreenwave_c30':
        mdl_map = lambda x: read_sumo_tsc_plan("sumowebster", scene_name, case, x, params=dict(cycle_length=30))
    elif map_mdl_name == 'sumogreenwave_c60':
        mdl_map = lambda x: read_sumo_tsc_plan("sumowebster", scene_name, case, x, params=dict(cycle_length=60))
    else:
        raise Exception('Unexpected baseline planner name')
    delays = model_delay_on_test_cases(scene_name, case, mdl_map, n_workers=n_workers, pool=pool, return_info=False,
                                       simulation_id=simulation_id)
    return delays


def get_optimizer_trajectory_best_delay(map_model_run_data):
    traj_best_delay = []
    for run_data_on_tau in map_model_run_data:
        traj_best_delay_on_tau = []
        for run_data_of_trial in run_data_on_tau:
            traj_best_delay_of_trial = [run_data_of_trial[0, -1]]
            for i in range(1, run_data_of_trial.shape[0]):
                traj_best_delay_of_trial.append(min(traj_best_delay_of_trial[-1], run_data_of_trial[i, -1]))
            traj_best_delay_on_tau.append(traj_best_delay_of_trial)
        traj_best_delay.append(traj_best_delay_on_tau)
    return traj_best_delay


def get_optimizer_best_delay(map_model_run_data):
    best_delays = []
    for run_data_on_tau in map_model_run_data:
        best_delays.append([np.min(run_data_of_trial[:, -1]) for run_data_of_trial in run_data_on_tau])
    return best_delays


def get_optimizer_early_stopping_nfe(map_model_run_data):
    n_stop_func_evals = []
    for run_data_on_tau in map_model_run_data:
        n_stop_func_evals.append([run_data_of_trial.shape[0] for run_data_of_trial in run_data_on_tau])
    return n_stop_func_evals


def get_lmm_performance_metric(map_model_name, scene_name, case, baseline_name='pso', time_window=900):
    from scipy.stats import wilcoxon
    baseline_data = utils.load_optim_based_map_model_data(baseline_name, scene_name, case)
    baseline_traj_delays = np.mean(np.array(get_optimizer_trajectory_best_delay(baseline_data)), axis=1)

    if map_model_name == 'lmm':
        map_model_best_delays = np.load(utils.get_mdl_result_path(scene_name, case, map_model_name))
        map_model_run_info = dict(wall_clock_time=0.0)  # nearly no time cost
        map_model_nfe = 0.0
    else:
        map_model_data = utils.load_optim_based_map_model_data(map_model_name, scene_name, case)
        map_model_run_info = utils.load_optim_based_map_model_run_info(map_model_name, scene_name, case)
        map_model_best_delays = np.mean(np.array(get_optimizer_best_delay(map_model_data)), axis=1)
        map_model_nfe = np.mean(np.array(get_optimizer_early_stopping_nfe(map_model_data)))

    better_count = None
    n_checkpts = baseline_traj_delays.shape[1]
    # print(np.arange(0, n_checkpts + 1, 50))
    for checkpts in np.arange(0, n_checkpts + 1, 50):
        id = max(0, checkpts - 1)
        if wilcoxon(map_model_best_delays, baseline_traj_delays[:, id]).pvalue < 0.05 and \
                np.mean(map_model_best_delays) < np.mean(baseline_traj_delays[:, id]):
            better_count = checkpts
        else:
            break
    perf_ratio = better_count / n_checkpts
    time_ratio = map_model_run_info['wall_clock_time'] / time_window
    nfe_ratio = map_model_nfe / n_checkpts
    # print(map_model_name, scene_name, case, f"{perf_ratio * 100:.1f}%",
    #       f"{time_ratio * 100:.1f}%",
    #       f"{nfe_ratio * 100:.1f}%",)
    return perf_ratio, time_ratio, nfe_ratio


if __name__ == "__main__":
    pass