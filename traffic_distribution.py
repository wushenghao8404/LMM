import numpy as np
import config
import os
import utils
scene_name = None
case = None
tlb = None
tub = None
tdim = None
taskid = None


def prepare(sn, ca):
    global scene_name, case, tlb, tub, tdim, taskid
    scene_name = sn
    case = ca
    tlb = config.config_dictionary[scene_name].MIN_FLOW_INPUT
    tub = config.config_dictionary[scene_name].MAX_FLOW_INPUT
    tdim = config.config_dictionary[scene_name].DIM_PROB_FEATURE
    taskid = 0


# noinspection PyTypeChecker
def enumerate_train_tau():
    global taskid
    if scene_name is None or case is None:
        raise Exception('Distribution is not initialized')
    taus = np.load(utils.get_train_tau_path(scene_name, case))
    if taskid >= taus.shape[0]:
        raise Exception('task id', taskid,'exceeds the storage')
        taskid = 0
    tau = taus[taskid, :]
    taskid = taskid + 1
    return tau


# noinspection PyTypeChecker
def enumerate_test_tau():
    global taskid
    if scene_name is None or case is None:
        raise Exception('Distribution is not initialized')
    taus = np.load(utils.get_test_tau_path(scene_name, case))
    if taskid >= taus.shape[0]:
        raise Exception('task id', taskid, 'exceeds the storage')
        taskid = taskid + 1
        return sample_tau_func()
    tau = taus[taskid, :]
    taskid = taskid + 1
    return tau


def sample_tau_func():
    if scene_name is None or case is None:
        raise Exception('Distribution is not initialized')

    if scene_name == "net_single1":
        # tdim = 2
        if case == 0:
            # 0 is main stream road
            # small variance distribution
            return np.clip(np.random.uniform([100, 10],
                                             [150, 50]), tlb, tub).astype(np.int)
        elif case == 1:
            # large variance distribution
            return np.clip(np.random.uniform([ 10, 10],
                                             [150, 150]), tlb, tub).astype(np.int)
        elif case == 2:
            # mixture-model distribution
            mode = np.random.choice(['all_busy', 'west_busy', 'north_busy', 'leisure'])
            if mode == 'all_busy':
                return np.clip(np.random.normal(loc=[125, 125],
                                                 scale=[20, 20]), tlb, tub).astype(np.int)
            elif mode == 'west_busy':
                return np.clip(np.random.normal(loc=[50, 125],
                                                scale=[20, 20]), tlb, tub).astype(np.int)
            elif mode == 'north_busy':
                return np.clip(np.random.normal(loc=[125, 50],
                                                scale=[20, 20]), tlb, tub).astype(np.int)
            elif mode == 'leisure':
                return np.clip(np.random.normal(loc=[50, 50],
                                                scale=[20, 20]), tlb, tub).astype(np.int)
        else:
            raise Exception("Invalid case number.")
    elif scene_name == "net_single2":
        # tdim = 4
        if case == 0:
            # 0, 3 are main stream roads
            # small variance distribution
            return np.clip(np.random.uniform([100, 10, 10, 100],
                                             [150, 50, 50, 150]), tlb, tub).astype(np.int)
        elif case == 1:
            # large variance distribution
            return np.clip(np.random.uniform([10, 10, 10, 10],
                                             [150, 50, 50, 150]), tlb, tub).astype(np.int)
        elif case == 2:
            # mixture-model distribution
            mode = np.random.choice(['all_busy', 'west_busy', 'east_busy', 'leisure'])
            if mode == 'all_busy':
                return np.clip(np.random.normal(loc=[125, 50, 50, 125],
                                                 scale=[20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'west_busy':
                return np.clip(np.random.normal(loc=[50, 50, 50, 125],
                                                scale=[20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'east_busy':
                return np.clip(np.random.normal(loc=[125, 50, 50, 50],
                                                scale=[20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'leisure':
                return np.clip(np.random.normal(loc=[50, 50, 50, 50],
                                                scale=[20, 20, 20, 20]), tlb, tub).astype(np.int)
        else:
            raise Exception("Invalid case number.")
    elif scene_name == "net_single3":
        # tdim = 4
        if case == 0:
            # 0, 3 are main stream roads
            # small variance distribution
            return np.clip(np.random.uniform([150,  10,  10, 150],
                                             [200,  50,  50, 200]), tlb, tub).astype(np.int)
        elif case == 1:
            # large variance distribution
            return np.clip(np.random.uniform([ 10,  10,  10,  10],
                                             [200,  50,  50, 200]), tlb, tub).astype(np.int)
        elif case == 2:
            # mixture-model distribution
            mode = np.random.choice(['all_busy', 'west_busy', 'east_busy', 'leisure'])
            if mode == 'all_busy':
                return  np.clip(np.random.normal(loc=[150, 50, 50, 150],
                                                 scale=[20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'west_busy':
                return np.clip(np.random.normal(loc=[50, 50, 50, 150],
                                                scale=[20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'east_busy':
                return np.clip(np.random.normal(loc=[150, 50, 50, 50],
                                                scale=[20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'leisure':
                return np.clip(np.random.normal(loc=[50, 50, 50, 50],
                                                scale=[20, 20, 20, 20]), tlb, tub).astype(np.int)
        else:
            raise Exception("Invalid case number.")
    elif scene_name == "net_double":
        # tdim = 6
        if case == 0:
            # 0, 5 are main stream roads
            # small variance distribution
            return np.clip(np.random.uniform([100, 10, 10, 10, 10, 100],
                                             [150, 50, 50, 50, 50, 150]), tlb, tub).astype(np.int)
        elif case == 1:
            # large variance distribution
            return np.clip(np.random.uniform([10,  10, 10, 10, 10, 10],
                                             [150, 50, 50, 50, 50, 150]), tlb, tub).astype(np.int)
        elif case == 2:
            # mixture-model distribution
            mode = np.random.choice(['all_busy', 'west_busy', 'east_busy', 'leisure'])
            if mode == 'all_busy':
                return  np.clip(np.random.normal(loc=[125, 50, 50, 50, 50, 125],
                                                 scale=[20, 20, 20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'west_busy':
                return np.clip(np.random.normal(loc=[125, 50, 50, 50, 50, 50],
                                                scale=[20, 20, 20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'east_busy':
                return np.clip(np.random.normal(loc=[50, 50, 50, 50, 50, 125],
                                                scale=[20, 20, 20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'leisure':
                return np.clip(np.random.normal(loc=[50, 50, 50, 50, 50, 50],
                                                scale=[20, 20, 20, 20, 20, 20]), tlb, tub).astype(np.int)
        else:
            raise Exception("Invalid case number.")
    elif scene_name == "net_2x2grid":
        # tdim = 8
        if case == 0:
            # 0, 1, 4, 5 are main stream roads
            # small variance distribution
            return np.clip(np.random.uniform([100, 100, 10, 10, 100, 100, 10, 10],
                                             [150, 150, 50, 50, 150, 150, 50, 50]), tlb, tub).astype(np.int)
        elif case == 1:
            # large variance distribution
            return np.clip(np.random.uniform([ 10,  10, 10, 10,  10,  10, 10, 10],
                                             [150, 150, 50, 50, 150, 150, 50, 50]), tlb, tub).astype(np.int)
        elif case == 2:
            # mixture-model distribution
            mode = np.random.choice(['all_busy', 'west_busy', 'east_busy', 'leisure'])
            if mode == 'all_busy':
                return  np.clip(np.random.normal(loc=[125, 125, 50, 50, 125, 125, 50, 50],
                                                 scale=[20, 20, 20, 20, 20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'west_busy':
                return np.clip(np.random.normal(loc=[125, 125, 50, 50, 50, 50, 50, 50],
                                                scale=[20, 20, 20, 20, 20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'east_busy':
                return np.clip(np.random.normal(loc=[50, 50, 50, 50, 125, 125, 50, 50],
                                                scale=[20, 20, 20, 20, 20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'leisure':
                return np.clip(np.random.normal(loc=[50, 50, 50, 50, 50, 50, 50, 50],
                                                scale=[20, 20, 20, 20, 20, 20, 20, 20]), tlb, tub).astype(np.int)
        else:
            raise Exception("Invalid case number.")
    elif scene_name == "net_3x3grid":
        # tdim = 12
        if case == 0:
            # 1, 3, 5, 6, 8, 10 are main stream roads
            # small variance distribution
            return np.clip(np.random.uniform([10, 100, 10, 100, 10, 100, 100, 10, 100,  10, 100, 10],
                                             [50, 150, 50, 150, 50, 150, 150, 50, 150,  50, 150, 50]), tlb, tub).astype(np.int)
        elif case == 1:
            # large variance distribution
            return np.clip(np.random.uniform([10,  10, 10,  10, 10,  10,  10, 10,  10,  10,  10, 10],
                                             [50, 150, 50, 150, 50, 150, 150, 50, 150,  50, 150, 50]), tlb, tub).astype(np.int)
        elif case == 2:
            # mixture-model distribution
            mode = np.random.choice(['all_busy', 'west_busy', 'east_busy', 'leisure'])
            if mode == 'all_busy':
                return np.clip(np.random.normal(loc=  [50, 125, 50, 125, 50, 125, 125, 50, 125,  50, 125, 50],
                                                scale=[20,  20, 20,  20, 20,  20,  20, 20,  20,  20,  20, 20]), tlb, tub).astype(np.int)
            elif mode == 'west_busy':
                # 1, 5, 8 busy
                return np.clip(np.random.normal(loc=  [50, 125, 50, 50, 50, 125, 50, 50, 125, 50, 50, 50],
                                                scale=[20,  20, 20, 20, 20,  20, 20, 20,  20, 20, 20, 20]), tlb, tub).astype(np.int)
            elif mode == 'east_busy':
                # 3, 6, 10 busy
                return np.clip(np.random.normal(loc=  [50, 50, 50, 125, 50, 50, 125, 50, 50, 50, 125, 50],
                                                scale=[20, 20, 20,  20, 20, 20,  20, 20, 20, 20,  20, 20]), tlb, tub).astype(np.int)
            elif mode == 'leisure':
                return np.clip(np.random.normal(loc=  [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
                                                scale=[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]), tlb, tub).astype(np.int)
        else:
            raise Exception("Invalid case number.")
    elif scene_name == "cologne1":
        # tdim = 5
        if case == 1:
            # 0, 4 are main stream roads
            return np.clip(np.random.uniform([50, 10,  50,  50,  10],
                                             [150, 50, 150, 150, 100]), tlb, tub).astype(np.int)
        else:
            raise Exception("Invalid case number.")
    elif scene_name == "cologne3":
        # tdim = 13
        if case == 1:
            # 1, 6 are main stream roads
            return np.clip(np.random.uniform([10,  50, 10, 10, 10, 10,  50, 10, 10, 10, 10, 10, 10],
                                             [50, 150, 50, 50, 50, 50, 150, 50, 50, 50, 50, 50, 50], ), tlb, tub).astype(np.int)
        else:
            raise Exception("Invalid case number.")
    elif scene_name == "ingolstadt1":
        # tdim = 4
        if case == 1:
            return np.clip(np.random.uniform([ 50,  50,  10,  50],
                                             [150, 150, 100, 150]), tlb, tub).astype(np.int)
        else:
            raise Exception("Invalid case number.")
    elif scene_name == "ingolstadt7":
        # tdim = 14
        if case == 1:
            # 3, 5, 10, 13 are main stream roads
            return np.clip(np.random.uniform([10, 10, 10,  50, 10,  50, 10, 10, 10, 10,  50, 10, 10,  50],
                                             [50, 50, 50, 150, 50, 150, 50, 50, 50, 50, 150, 50, 50, 150]), tlb,
                           tub).astype(np.int)
        else:
            raise Exception("Invalid case number.")
    else:
        raise Exception("Invalid scene name.")
