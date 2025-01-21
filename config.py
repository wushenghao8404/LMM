import numpy as np
import pandas as pd
import sumolib
import os
PROJECT_PATH = "/public2/home/wushenghao/project/LMM/"
SUMO_PATH = "/public2/home/wushenghao/anaconda3/envs/traffic/lib/python3.6/site-packages/sumo/"

# coll_alg_names = ['sur', 'grid', 'es', 'es_tr1', 'es_tr2', 'es_tr3', 'es_tr4', 'es_tr5',]
coll_alg_names = ['spso', 'stpso', 'mtpso']
pprc_alg_names = ['wsr', 'msr', 'ssr', 'ce']
optimizer_names = ['']
# data collector algorithm configuration
terminate_cond = "maxntau"
maxstag = 200
dc_cfgs = {
    'spso': {
        'early_stop': False,
    },
    'stpso':{
        'early_stop': True,
    },
    'mtpso':{
        'early_stop': True,
        'num_trsols': 20,
    },
}

# post-processing algorithm configuration
pp_ntau = 1000
pp_cfgs = {
    'pp9': {
        'n_probs'      : 100,
    },
    'ce': {
        'n_probs'      : 100,
    },
}

# final model selection configuration
fm_cfg = {
    'candidate_mdl':[
        {
            'alg_name': 'ssr',
            'planner_name': 'ss',
            'coll_alg_name': 'mtpso',
            'run_id': 0,
        },
        {
            'alg_name': 'msr',
            'planner_name': 'knn',
            'coll_alg_name': 'mtpso',
            'run_id': 0,
        },
        {
            'alg_name': 'wsr',
            'planner_name': 'knn',
            'coll_alg_name': 'mtpso',
            'run_id': 0,
        },
        {
            'alg_name': 'mtpso',
            'planner_name': 'knn',
            'run_id': 0,
            'data_size': None,
        },
    ]
}

# optimizer config
hyperparam_DE = {
    'popsize'  : 50,
    'maxnfe'   : 1000,
    'maxstag'  : 1e10,
    'mut_param': {
        'scheme': 'rand/1', # rand/1 or best/1
        'f'     : 0.5,
        'f_type': 'fix',  # fix or adaptive
    },
    'crs_param': {
        'scheme' : 'binomial',
        'cr'     : 0.9,
        'cr_type': 'fix',  # fix or adaptive
    },
    'sel_param': {
        'scheme' : 'elite', # elite or crowd
    }
}

optimizer_hp = {
    'TrPSO':
        {
            'num_trsols': 20,
        },
    'REPSO':
        {
            'num_trsols': 1,
        },
    'DualTrPSO':
        {
            'pw': 0.0,
            'pn': 1.0,
        }
}

# display config
display_bar_len = 40

# baseline tsc default setting
default_baseline_tsc_hp = {
    'webster':
        {
            'update_freq': 10,
        },
    'sotl':
        {
            'theta': 45,
            'omega': 1,
            'mu': 3,
        },
}

# problem config
scene_names = ['net_single1',
               'net_single2',
               'net_single3',
               'net_double' ,
               'net_2x2grid',
               'net_3x3grid',
               'cologne1'   ,
               'cologne3'   ,
               'ingolstadt1',
               'ingolstadt7',]

scene_cases = {'net_single1' : [0, 1, 2],
               'net_single2': [0, 1, 2],
               'net_single3': [0, 1, 2],
               'net_double': [0, 1, 2],
               'net_2x2grid': [0, 1, 2],
               'net_3x3grid' : [0, 1, 2],
               'cologne1'    : [1,],
               'cologne3'    : [1,],
               'ingolstadt1' : [1,],
               'ingolstadt7' : [1,],}


search_fes={'net_single1' : 500,
            'net_single2' : 1000,
            'net_single3' : 1000,
            'net_double'  : 1000,
            'net_2x2grid' : 1000,
            'net_3x3grid' : 1000,
            'cologne1'    : 1000,
            'cologne3'    : 1000,
            'ingolstadt1' : 1000,
            'ingolstadt7' : 1000,}


sample_budget ={'net_single1' :  500 * 1000,
                'net_single2' : 1000 * 1000,
                'net_single3' : 1000 * 1000,
                'net_double'  : 1000 * 1000,
                'net_2x2grid' : 1000 * 1000,
                'net_3x3grid' : 1000 * 1000,
                'cologne1'    : 1000 * 1000,
                'cologne3'    : 1000 * 1000,
                'ingolstadt1' : 1000 * 1000,
                'ingolstadt7' : 1000 * 1000,}


class CONFIG:
    def __init__(self, SCENE_NAME, DIM_DECVAR, DIM_PROB_FEATURE, MIN_FLOW_INPUT, MAX_FLOW_INPUT ,
                 MIN_OFFSET, MAX_OFFSET, MIN_GREEN_DURATION, MAX_GREEN_DURATION, MIN_CYCLE_TIME, MAX_CYCLE_TIME):
        global PROJECT_PATH
        self.SCENE_NAME = SCENE_NAME
        self.NET_PATH = os.path.join(PROJECT_PATH, "data", "net", self.SCENE_NAME + ".net.xml")
        
        if DIM_DECVAR is None:
            self.DIM_DECVAR = self.get_xdim()
        else:
            self.DIM_DECVAR = DIM_DECVAR
        
        if DIM_PROB_FEATURE is None:
            self.DIM_PROB_FEATURE = self.get_tdim()
        else:
            self.DIM_PROB_FEATURE = DIM_PROB_FEATURE
        
        self.MIN_FLOW_INPUT = MIN_FLOW_INPUT * np.ones(self.DIM_PROB_FEATURE) \
            if not isinstance(MIN_FLOW_INPUT, np.ndarray) else MIN_FLOW_INPUT
        self.MAX_FLOW_INPUT = MAX_FLOW_INPUT * np.ones(self.DIM_PROB_FEATURE) \
            if not isinstance(MAX_FLOW_INPUT, np.ndarray) else MAX_FLOW_INPUT
        self.NUM_OFFSET = self.get_n_offset()
        self.NUM_DURATION = self.get_n_duration()
        self.MIN_OFFSET = MIN_OFFSET * np.ones(self.NUM_OFFSET) \
            if not isinstance(MIN_OFFSET, np.ndarray) else MIN_OFFSET
        self.MAX_OFFSET = MAX_OFFSET * np.ones(self.NUM_OFFSET) \
            if not isinstance(MAX_OFFSET, np.ndarray) else MAX_OFFSET
        self.MIN_GREEN_DURATION = MIN_GREEN_DURATION * np.ones(self.NUM_DURATION) \
            if not isinstance(MIN_GREEN_DURATION, np.ndarray) else MIN_GREEN_DURATION
        self.MAX_GREEN_DURATION = MAX_GREEN_DURATION * np.ones(self.NUM_DURATION) \
            if not isinstance(MAX_GREEN_DURATION, np.ndarray) else MAX_GREEN_DURATION
        self.MIN_CYCLE_TIME = MIN_CYCLE_TIME
        self.MAX_CYCLE_TIME = MAX_CYCLE_TIME
        self.load_net_param()

    def load_net_param(self):
        net = sumolib.net.readNet(self.NET_PATH, withPrograms=True)
        tlsList = net.getTrafficLights()
        self.n_phases = []
        self.ts_ids = []
        self.ts_phase_state = {}
        for tls in tlsList:
            tls_id = tls.getID()
            self.ts_phase_state[tls_id] = []
            programs = tls.getPrograms()
            n_green_phases = 0
            if len(programs) > 1:
                raise Exception("The number of existing tls program is more than 1")
            program = programs[list(programs.keys())[0]]
            phases = program.getPhases()
            for phase in phases:
                if 'y' not in phase.state and ('G' in phase.state or 'g' in phase.state):
                    n_green_phases += 1
                self.ts_phase_state[tls_id].append(phase.state)
            self.n_phases.append(n_green_phases)
            self.ts_ids.append(tls_id)
        # print(self.ts_phase_state)

    def get_sim_scene_dir(self):
        return os.path.join(PROJECT_PATH, "sim", self.SCENE_NAME)

    def get_tdim(self):
        net = sumolib.net.readNet(self.NET_PATH)
        edgeList = net.getEdges()
        tdim = 0
        for edge in edgeList:
            if edge.is_fringe() and len(edge.getIncoming()) == 0:
                tdim += 1
        return tdim
    
    def get_tlb(self):
        return self.MIN_FLOW_INPUT
        
    def get_tub(self):
        return self.MAX_FLOW_INPUT

    def get_n_duration(self):
        net = sumolib.net.readNet(self.NET_PATH, withPrograms=True)
        tlsList = net.getTrafficLights()
        n_dur = 0
        for tls in tlsList:
            programs = tls.getPrograms()
            n_green_phases = 0
            if len(programs) > 1:
                raise Exception("The number of existing tls program is more than 1")
            program = programs[list(programs.keys())[0]]
            phases = program.getPhases()
            for phase in phases:
                if 'y' not in phase.state and ('G' in phase.state or 'g' in phase.state):
                    n_green_phases += 1
            n_dur += n_green_phases
        return n_dur

    def get_n_offset(self):
        net = sumolib.net.readNet(self.NET_PATH, withPrograms=True)
        tlsList = net.getTrafficLights()
        n_off = 0
        if len(tlsList) > 1:
            n_off = len(tlsList)
        return n_off

    def get_xdim(self):
        net = sumolib.net.readNet(self.NET_PATH, withPrograms=True)
        tlsList = net.getTrafficLights()
        xdim = self.get_n_duration()
        if len(tlsList) > 1:
            xdim += len(tlsList)
        return xdim

    def get_xlb(self):
        if self.NUM_OFFSET > 1:
            return np.r_[self.MIN_GREEN_DURATION, self.MIN_OFFSET]
        else:
            return self.MIN_GREEN_DURATION
    
    def get_xub(self):
        if self.NUM_OFFSET > 1:
            return np.r_[self.MAX_GREEN_DURATION, self.MAX_OFFSET]
        else:
            return self.MAX_GREEN_DURATION


'''configuration dictionary'''      
config_dictionary = {
        'net_single1' : CONFIG( SCENE_NAME="net_single1",
                                DIM_DECVAR=2, DIM_PROB_FEATURE=2, MIN_FLOW_INPUT=10, MAX_FLOW_INPUT=200,
                                MIN_OFFSET=-60, MAX_OFFSET=60,
                                MIN_GREEN_DURATION=5, MAX_GREEN_DURATION=30, MIN_CYCLE_TIME=25, MAX_CYCLE_TIME=120),
        'net_single2' : CONFIG( SCENE_NAME="net_single2",
                                DIM_DECVAR=4, DIM_PROB_FEATURE=4, MIN_FLOW_INPUT=10, MAX_FLOW_INPUT=200,
                                MIN_OFFSET=-60, MAX_OFFSET=60,
                                MIN_GREEN_DURATION=5, MAX_GREEN_DURATION=30, MIN_CYCLE_TIME=25, MAX_CYCLE_TIME=120),
        'net_single3' : CONFIG( SCENE_NAME="net_single3",
                                DIM_DECVAR=4, DIM_PROB_FEATURE=4, MIN_FLOW_INPUT=10, MAX_FLOW_INPUT=200,
                                MIN_OFFSET=-60, MAX_OFFSET=60,
                                MIN_GREEN_DURATION=5, MAX_GREEN_DURATION=30, MIN_CYCLE_TIME=25, MAX_CYCLE_TIME=120),
        'net_double'  : CONFIG( SCENE_NAME="net_double",
                                DIM_DECVAR=10, DIM_PROB_FEATURE=6, MIN_FLOW_INPUT=10, MAX_FLOW_INPUT=200,
                                MIN_OFFSET=-60, MAX_OFFSET=60,
                                MIN_GREEN_DURATION=5, MAX_GREEN_DURATION=15, MIN_CYCLE_TIME=25, MAX_CYCLE_TIME=120),
        'net_2x2grid' : CONFIG( SCENE_NAME="net_2x2grid",
                                DIM_DECVAR=20, DIM_PROB_FEATURE=8, MIN_FLOW_INPUT=10, MAX_FLOW_INPUT=200,
                                MIN_OFFSET=-60, MAX_OFFSET=60,
                                MIN_GREEN_DURATION=5, MAX_GREEN_DURATION=15, MIN_CYCLE_TIME=25, MAX_CYCLE_TIME=120),
        'net_3x3grid' : CONFIG( SCENE_NAME="net_3x3grid",
                                DIM_DECVAR=27, DIM_PROB_FEATURE=12, MIN_FLOW_INPUT=10, MAX_FLOW_INPUT=200,
                                MIN_OFFSET=-60, MAX_OFFSET=60,
                                MIN_GREEN_DURATION=5, MAX_GREEN_DURATION=15, MIN_CYCLE_TIME=25, MAX_CYCLE_TIME=120),
        'cologne1'    : CONFIG( SCENE_NAME="cologne1",
                                DIM_DECVAR=None, DIM_PROB_FEATURE=None, MIN_FLOW_INPUT=10, MAX_FLOW_INPUT=300,
                                MIN_OFFSET=-60, MAX_OFFSET=60,
                                MIN_GREEN_DURATION=5, MAX_GREEN_DURATION=30, MIN_CYCLE_TIME=25, MAX_CYCLE_TIME=120),
        'cologne3'    : CONFIG( SCENE_NAME="cologne3",
                                DIM_DECVAR=None, DIM_PROB_FEATURE=None, MIN_FLOW_INPUT=10, MAX_FLOW_INPUT=300,
                                MIN_OFFSET=-60, MAX_OFFSET=60,
                                MIN_GREEN_DURATION=5, MAX_GREEN_DURATION=15, MIN_CYCLE_TIME=25, MAX_CYCLE_TIME=120),
        'ingolstadt1' : CONFIG( SCENE_NAME="ingolstadt1",
                                DIM_DECVAR=None, DIM_PROB_FEATURE=None, MIN_FLOW_INPUT=10, MAX_FLOW_INPUT=300,
                                MIN_OFFSET=-60, MAX_OFFSET=60,
                                MIN_GREEN_DURATION=5, MAX_GREEN_DURATION=30, MIN_CYCLE_TIME=25, MAX_CYCLE_TIME=120),
        'ingolstadt7' : CONFIG( SCENE_NAME="ingolstadt7",
                                DIM_DECVAR=None, DIM_PROB_FEATURE=None, MIN_FLOW_INPUT=10, MAX_FLOW_INPUT=300,
                                MIN_OFFSET=-60, MAX_OFFSET=60,
                                MIN_GREEN_DURATION=5, MAX_GREEN_DURATION=15, MIN_CYCLE_TIME=25, MAX_CYCLE_TIME=120),
        }

task_dictionary = {'net_single1': [0, 1, 2],
                   'net_single2': [0, 1, 2],
                   'net_single3': [0, 1, 2],
                   'net_double': [0, 1, 2],
                   'net_2x2grid': [0, 1, 2],
                   'net_3x3grid': [0, 1, 2],
                   'cologne1': [1],
                   'cologne3': [1],
                   'ingolstadt1': [1],
                   'ingolstadt7': [1],
                   }


def save_cfg():
    cfg_dict = {'name': pd.Series([key for key in config_dictionary.keys()]),
                'case': pd.Series([str(scene_cases[key])[1:-1] for key in config_dictionary.keys()]),
                'tdim': pd.Series([config_dictionary[key].get_tdim() for key in config_dictionary.keys()]),
                'tlb' : pd.Series([config_dictionary[key].get_tlb()[0] for key in config_dictionary.keys()]),
                'tub' : pd.Series([config_dictionary[key].get_tub()[0] for key in config_dictionary.keys()]),
                'xdim': pd.Series([config_dictionary[key].get_xdim() for key in config_dictionary.keys()]),
                'xlb': pd.Series([config_dictionary[key].get_xlb()[0] for key in config_dictionary.keys()]),
                'xub': pd.Series([config_dictionary[key].get_xub()[0] for key in config_dictionary.keys()]),
                'search_fes': pd.Series([str(search_fes[key]) for key in config_dictionary.keys()]),
                'sample_budget': pd.Series([str(sample_budget[key]) for key in config_dictionary.keys()]), }
    columns = ['name', 'case', 'tdim', 'tlb', 'tub', 'xdim', 'xlb', 'xub', 'search_fes', 'sample_budget']
    df = pd.DataFrame(cfg_dict)[columns]
    print(df)
    df.to_excel(PROJECT_PATH + 'config.xlsx')


if __name__ == "__main__":
    save_cfg()