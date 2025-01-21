import numpy as np
import os
import utils
import config
import traffic_problem
import optimizer
import pickle
import copy
import torch
from torch.autograd import Variable
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
from sklearn.decomposition import PCA
import traffic_optimizer

model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
model_LinearRegression = linear_model.LinearRegression()
model_SVR = svm.SVR()
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)
kernel = RBF(length_scale=10)
model_GaussianProcessRegressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)


class Net(torch.nn.Module):
    def __init__(self, n_feature=1, n_hidden=10, n_output=1):
        # 初始网络的内部结构
        super().__init__()
        self.nn = torch.nn.Sequential(torch.nn.Linear(n_feature, n_hidden),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(n_hidden, n_output)
                                     )

    def forward(self, x):
        # 一次正向行走过程
        return self.nn(x)
    
    def fit(self, train_x, train_y):
        input_dim = train_x.shape[1]
        output_dim = train_y.shape[1]
        n_hidden = 128
        self.__init__(n_feature=input_dim, n_hidden=n_hidden, n_output=output_dim)

        batch_size = train_x.shape[0]
        nbatch = train_x.shape[0] // batch_size
        # print(train_x.shape)
        x, y = copy.deepcopy(train_x), copy.deepcopy(train_y)
        x, y = Variable(torch.tensor(x).float()), Variable(torch.tensor(y).float())
        loss_func = torch.nn.MSELoss()
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1, weight_decay=1e-4)
        epoch = 0
        while epoch < 1000:
            shuf_index = np.arange(train_x.shape[0])
            np.random.shuffle(shuf_index)
            for i in range(nbatch):
                batch_index = shuf_index[i*batch_size:(i+1)*batch_size]
                batch_x = x[batch_index,:]
                batch_y = y[batch_index,:]
                prediction = self(batch_x)
                loss = loss_func(prediction, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 50 == 0:
                print("\rEpoch:{} , Loss:{}".format(epoch, loss.data),end="")
            epoch += 1
        print("\rEpoch:{} , Loss:{}".format(epoch, loss.data))
        return self
            
    def predict(self,x):
        return self.forward(Variable(torch.tensor(x)).float()).detach().numpy()


class LinearNet(torch.nn.Module):
    def __init__(self, n_feature=1, n_hidden=10, n_output=1):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(n_feature, n_hidden),
            torch.nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        return self.nn(x)
    
    def fit(self, train_x, train_y):
        input_dim = train_x.shape[1]
        output_dim = train_y.shape[1]
        n_hidden = 128
        self.__init__(n_feature=input_dim, n_hidden=n_hidden, n_output=output_dim)

        x, y = copy.deepcopy(train_x), copy.deepcopy(train_y)
        x, y = Variable(torch.tensor(x).float()), Variable(torch.tensor(y).float())
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        epoch = 0
        while epoch < 1000:
            prediction = self(x)
            loss = loss_func(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print("\rEpoch:{} , Loss:{}".format(epoch, loss.data),end="")
            epoch += 1
        print("\rEpoch:{} , Loss:{}".format(epoch, loss.data))
        return self
            
    def predict(self,x):
        return self.forward(Variable(torch.tensor(x)).float()).detach().numpy()


model_NeuralNet = Net()


class LearningBasedMap:
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, verbose=0, data_path=None, data_size=None):
        self.scene_name = scene_name
        self.case = case
        self.cfg = config.config_dictionary[self.scene_name]
        self.xlb = self.cfg.get_xlb()
        self.xub = self.cfg.get_xub()
        self.tlb = self.cfg.get_tlb()
        self.tub = self.cfg.get_tub()
        self.coll_algo_name = coll_algo_name
        self.coll_run_id = coll_run_id
        self.data_path = data_path
        if self.data_path is None:
            self.data_path = utils.get_collected_dataset_path(self.coll_algo_name, self.scene_name, self.case, self.coll_run_id)
        self.dataset = np.load(self.data_path, allow_pickle=True)
        if data_size is not None:
            if data_size > self.dataset.shape[0]:
                raise Exception("The input data_size exceeds the size of training data.")
            self.dataset = self.dataset[:data_size, :]
        self.data_tau = self.dataset[:, :self.cfg.DIM_PROB_FEATURE]
        self.data_xopt = self.dataset[:, self.cfg.DIM_PROB_FEATURE:]
        self.verbose = verbose


class LinearMap(LearningBasedMap):
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, verbose=0, data_path=None, data_size=None):
        super().__init__(scene_name=scene_name, case=case, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)
        self.lr = model_LinearRegression.fit(self.data_tau, self.data_xopt)
        self.map_name = 'lr'

    def map(self, tau):
        return np.clip(self.lr.predict(tau.reshape(1,-1)).flatten().astype(np.int), self.xlb, self.xub)


class NeuralMap(LearningBasedMap):
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, verbose=0, data_path=None, data_size=None):
        super().__init__(scene_name=scene_name, case=case, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)
        self.lr = model_NeuralNet.fit(self.data_tau, self.data_xopt)
        self.map_name = 'nn'

    def map(self, tau):
        return np.clip(self.lr.predict(tau.reshape(1, -1)).flatten().astype(np.int), self.xlb, self.xub)


class DecisionTreeMap(LearningBasedMap):
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, verbose=0, data_path=None, data_size=None):
        super().__init__(scene_name=scene_name, case=case, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)
        self.lr = model_DecisionTreeRegressor.fit(self.data_tau, self.data_xopt)
        self.map_name = 'dt'

    def map(self, tau):
        return np.clip(self.lr.predict(tau.reshape(1,-1)).flatten().astype(np.int), self.xlb, self.xub)


class RandomForestMap(LearningBasedMap):
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, verbose=0, data_path=None, data_size=None):
        super().__init__(scene_name=scene_name, case=case, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)
        self.lr = model_RandomForestRegressor.fit(self.data_tau, self.data_xopt)
        self.map_name = 'rf'

    def map(self, tau):
        return np.clip(self.lr.predict(tau.reshape(1,-1)).flatten().astype(np.int), self.xlb, self.xub)


class GaussianSample(LearningBasedMap):
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, mu=None, sigma=None, verbose=0, data_path=None,
                 data_size=None):
        super().__init__(scene_name=scene_name, case=case, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)
        if mu and sigma:
            self.mu = mu
            self.sigma = sigma
        else:
            self.learn()
        self.map_name = 'gs'

    def learn(self):
        self.mu = np.mean(self.data_xopt, axis=0)
        self.sigma = np.std(self.data_xopt, axis=0)

    def map(self, tau):
        return np.clip(np.random.randn(self.cfg.DIM_DECVAR) * self.sigma + self.mu, self.xlb, self.xub)


class MeanOfSample(LearningBasedMap):
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, verbose=0, data_path=None, data_size=None):
        super().__init__(scene_name=scene_name, case=case, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)
        self.mu = np.mean(self.data_xopt, axis=0)
        self.map_name = 'mos'

    def map(self, tau):
        return self.mu


class RandomSelection(LearningBasedMap):
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, verbose=0, data_path=None, data_size=None):
        super().__init__(scene_name=scene_name, case=case, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)
        self.map_name = 'rs'

    def map(self, tau):
        return self.data_xopt[np.random.randint(self.data_xopt.shape[0]),:]


class NearestNeighborOffline(LearningBasedMap):
    def __init__(self, scene_name, case, coll_algo_name,
                 distance_metric='L1', n_neighbors=1, coll_run_id=0, verbose=0, data_path=None, data_size=None):
        super().__init__(scene_name=scene_name, case=case, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)
        self.distance_metric = distance_metric
        self.n_neighbors = n_neighbors
        if self.distance_metric not in ['L1', 'L2', 'Linf', 'pearson_corr', 'PCA1', 'PCA2']:
            raise Exception('Unexpected distance metric.')
        if self.distance_metric == 'PCA1' and self.data_tau is not None:
            self.pca = PCA(n_components=1)
            self.pca.fit(self.data_tau)
            self.t_data_tau = self.pca.transform((self.data_tau - self.tlb) / (self.tub - self.tlb))
        elif self.distance_metric == 'PCA2' and self.data_tau is not None:
            self.pca = PCA(n_components=2)
            self.pca.fit(self.data_tau)
            self.t_data_tau = self.pca.transform((self.data_tau - self.tlb) / (self.tub - self.tlb))
        if distance_metric != 'L1' and n_neighbors > 1:
            raise Exception('Distance metric other than L1 does not support K > 1')
        if self.verbose == 1:
            print("mean tau", np.mean(self.data_tau, axis=0))
            print("std tau", np.std(self.data_tau, axis=0))
            print("mean xopt", np.mean(self.data_xopt, axis=0))
            print("std xopt", np.std(self.data_xopt, axis=0))
        self.map_name = 'knn_off'

    def fit(self, data_tau, data_xopt):
        if data_tau.shape[0] != data_xopt.shape[0]:
            raise Exception("Size Distmatched: Each instance data should have a corresponding Y or label")
        if data_tau.shape[1] != self.cfg.DIM_PROB_FEATURE:
            raise Exception("Size Distmatched: Input data_tau dimension does not match task feature dimension")
        if data_xopt.shape[1] != self.cfg.DIM_DECVAR:
            raise Exception("Size Distmatched: Input data_xopt dimension does not match solution feature dimension")
        self.data_tau = data_tau
        self.data_xopt = data_xopt
        if self.distance_metric == 'PCA1':
            self.pca = PCA(n_components=1)
            self.pca.fit(self.data_tau)
            self.t_data_tau = self.pca.transform((self.data_tau - self.tlb) / (self.tub - self.tlb))
        elif self.distance_metric == 'PCA2':
            self.pca = PCA(n_components=2)
            self.pca.fit(self.data_tau)
            self.t_data_tau = self.pca.transform((self.data_tau - self.tlb) / (self.tub - self.tlb))
        self.dataset = np.c_[self.data_tau, self.data_xopt]

    def calculate_stats(self):
        ntau = self.data_tau.shape[0]
        tdists = np.ones((ntau, ntau)) * 1e10
        for i in range(ntau):
            for j in range(i+1, ntau):
                tdists[i,j] = np.sum(np.abs(self.data_tau[i, :] - self.data_tau[j, :]))
                tdists[j,i] = tdists[i,j]
        nis = [np.argmin(tdists[i,:]) for i in range(ntau)]
        lfs = np.array([np.sum(np.abs(self.data_xopt[i,:] - self.data_xopt[nis[i],:])) /
                        (np.sum(np.abs(self.data_tau[i,:] - self.data_tau[nis[i],:])) + 1e-10) for i in range(ntau)])
        self.data_Lf = lfs

    def predict(self, tau):
        if self.distance_metric == 'L1':
            dists = np.array([np.sum(np.abs(tau - tau0)) for tau0 in self.data_tau]) + 1e-10
        elif self.distance_metric == 'L2':
            dists = np.array([np.sqrt(np.sum((tau - tau0) ** 2)) for tau0 in self.data_tau]) + 1e-10
        elif self.distance_metric == 'Linf':
            dists = np.array([np.max(np.abs(tau - tau0)) for tau0 in self.data_tau]) + 1e-10
        elif self.distance_metric == 'pearson_corr':
            dists = []
            for tau0 in self.data_tau:
                if np.sum(np.abs(tau - tau0)) < 1.0:
                    dists.append(1e-10)
                else:
                    dists.append(1 - np.abs(np.corrcoef(tau + np.random.random(len(tau)) * 0.001,
                                                        tau0 + np.random.random(len(tau)) * 0.001)[0, 1]) + 1e-10)
            return np.clip(self.data_xopt[np.argmin(dists), :], self.xlb, self.xub).astype(np.int)
        elif self.distance_metric == 'PCA1' or self.distance_metric == 'PCA2':
            n_tau = ((tau - self.tlb) / (self.tub- self.tlb)).reshape(1,-1)
            t_tau = self.pca.transform(n_tau)
            dists = np.array([np.sqrt(np.sum((t_tau[0,:] - t_tau0) ** 2)) for t_tau0 in self.t_data_tau]) + 1e-10
        sorted_index = np.argsort(dists)
        predx = np.clip(np.mean(self.data_xopt[sorted_index[:self.n_neighbors], :], axis=0), self.xlb, self.xub).astype(np.int)
        rank = np.argsort(np.argsort(self.data_Lf))
        return predx, sorted_index[:self.n_neighbors], self.data_Lf[sorted_index[:self.n_neighbors]], rank[sorted_index[:self.n_neighbors]]

    def map(self, tau):
        dists = None
        if self.distance_metric == 'L1':
            dists = np.array([np.sum(np.abs(tau - tau0)) for tau0 in self.data_tau]) + 1e-10
        elif self.distance_metric == 'L2':
            dists = np.array([np.sqrt(np.sum((tau - tau0) ** 2)) for tau0 in self.data_tau]) + 1e-10
        elif self.distance_metric == 'Linf':
            dists = np.array([np.max(np.abs(tau - tau0)) for tau0 in self.data_tau]) + 1e-10
        elif self.distance_metric == 'pearson_corr':
            dists = []
            for tau0 in self.data_tau:
                if np.sum(np.abs(tau - tau0)) < 1.0:
                    dists.append(1e-10)
                else:
                    dists.append(1 - np.abs(np.corrcoef(tau + np.random.random(len(tau)) * 0.001,
                                                        tau0 + np.random.random(len(tau)) * 0.001)[0, 1]) + 1e-10)
            return np.clip(self.data_xopt[np.argmin(dists), :], self.xlb, self.xub).astype(np.int)
        elif self.distance_metric == 'PCA1' or self.distance_metric == 'PCA2':
            n_tau = ((tau - self.tlb) / (self.tub - self.tlb)).reshape(1, -1)
            t_tau = self.pca.transform(n_tau)
            dists = np.array([np.sqrt(np.sum((t_tau[0, :] - t_tau0) ** 2)) for t_tau0 in self.t_data_tau]) + 1e-10
        index = np.argsort(dists)
        # if self.verbose == 1:
        #     print("tau to datatau minimum distance", np.min(dists))
        return np.clip(np.mean(self.data_xopt[index[:self.n_neighbors],:],axis=0), self.xlb, self.xub).astype(np.int)


class NearestNeighborOnline:
    def __init__(self, scene_name, case, distance_metric='L1', n_neighbors=1, data_tau=None, data_xopt=None):
        self.scene_name = scene_name
        self.case = case
        self.cfg = config.config_dictionary[self.scene_name]
        self.xlb = self.cfg.get_xlb()
        self.xub = self.cfg.get_xub()
        self.tlb = self.cfg.get_tlb()
        self.tub = self.cfg.get_tub()
        if distance_metric not in ['L1', 'L2', 'Linf', 'pearson_corr', 'PCA1', 'PCA2']:
            raise Exception('Unexpected distance metric.')
        self.distance_metric = distance_metric
        self.n_neighbors = n_neighbors
        self.data_tau = data_tau
        self.data_xopt = data_xopt
        if isinstance(self.data_tau, list):
            self.data_tau = np.array(self.data_tau)
        if isinstance(self.data_xopt, list):
            self.data_xopt = np.array(self.data_xopt)
        if self.distance_metric == 'PCA1' and self.data_tau is not None:
            self.pca = PCA(n_components=1)
            self.pca.fit(self.data_tau)
            self.t_data_tau = self.pca.transform((self.data_tau - self.tlb) / (self.tub- self.tlb))
        elif self.distance_metric == 'PCA2' and self.data_tau is not None:
            self.pca = PCA(n_components=2)
            self.pca.fit(self.data_tau)
            self.t_data_tau = self.pca.transform((self.data_tau - self.tlb) / (self.tub- self.tlb))
        if distance_metric != 'L1' and n_neighbors > 1:
            raise Exception('Distance metric other than L1 does not support K > 1')
        self.map_name = 'knn_on'

    def add_data(self, tau, xopt):
        if self.data_tau is None:
            self.data_tau = np.array([tau])
            self.data_xopt = np.array([xopt])
        else:
            self.data_tau = np.r_[self.data_tau, tau.reshape(1,-1)]
            self.data_xopt = np.r_[self.data_xopt, xopt.reshape(1,-1)]
        if self.distance_metric == 'PCA1':
            self.pca = PCA(n_components=1)
            self.pca.fit(self.data_tau)
            self.t_data_tau = self.pca.transform((self.data_tau - self.tlb) / (self.tub- self.tlb))
        elif self.distance_metric == 'PCA2':
            self.pca = PCA(n_components=2)
            self.pca.fit(self.data_tau)
            self.t_data_tau = self.pca.transform((self.data_tau - self.tlb) / (self.tub- self.tlb))

    def fit(self, data_tau, data_xopt):
        self.data_tau = data_tau
        self.data_xopt = data_xopt
        if isinstance(self.data_tau, list):
            self.data_tau = np.array(self.data_tau)
        if isinstance(self.data_xopt, list):
            self.data_xopt = np.array(self.data_xopt)
        if self.distance_metric == 'PCA1':
            self.pca = PCA(n_components=1)
            self.pca.fit(self.data_tau)
            self.t_data_tau = self.pca.transform((self.data_tau - self.tlb) / (self.tub- self.tlb))
        elif self.distance_metric == 'PCA2':
            self.pca = PCA(n_components=2)
            self.pca.fit(self.data_tau)
            self.t_data_tau = self.pca.transform((self.data_tau - self.tlb) / (self.tub- self.tlb))

    def map(self, tau, return_index=False):
        if self.data_tau is None or self.data_xopt is None:
            print('The support dataset is empty, return random solution as output')
            return np.random.uniform(self.xlb, self.xub).astype(np.int)
        if self.distance_metric == 'L1':
            dists = np.array([np.sum(np.abs(tau - tau0)) for tau0 in self.data_tau]) + 1e-10
        elif self.distance_metric == 'L2':
            dists = np.array([np.sqrt(np.sum((tau - tau0) ** 2)) for tau0 in self.data_tau]) + 1e-10
        elif self.distance_metric == 'Linf':
            dists = np.array([np.max(np.abs(tau - tau0)) for tau0 in self.data_tau]) + 1e-10
        elif self.distance_metric == 'pearson_corr':
            dists = []
            for tau0 in self.data_tau:
                if np.sum(np.abs(tau - tau0)) < 1.0:
                    dists.append(1e-10)
                else:
                    dists.append(1 - np.abs(np.corrcoef(tau + np.random.random(len(tau)) * 0.001,
                                                        tau0 + np.random.random(len(tau)) * 0.001)[0, 1]) + 1e-10)
            return np.clip(self.data_xopt[np.argmin(dists), :], self.xlb, self.xub).astype(np.int)
        elif self.distance_metric == 'PCA1' or self.distance_metric == 'PCA2':
            n_tau = ((tau - self.tlb) / (self.tub - self.tlb)).reshape(1, -1)
            t_tau = self.pca.transform(n_tau)
            dists = np.array([np.sqrt(np.sum((t_tau[0, :] - t_tau0) ** 2)) for t_tau0 in self.t_data_tau]) + 1e-10
        indices = np.argsort(dists)[:self.n_neighbors]
        weights = dists[indices] / np.sum(dists[indices])
        if not return_index:
            return np.clip(np.average(self.data_xopt[indices, :], weights=weights, axis=0), self.xlb, self.xub).astype(np.int)
        else:
            return np.clip(np.average(self.data_xopt[indices, :], weights=weights, axis=0), self.xlb, self.xub).astype(
                np.int), indices


class UniformSample(LearningBasedMap):
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, verbose=0, data_path=None, data_size=None):
        super().__init__(scene_name=scene_name, case=case, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)
        self.map_name = 'us'
        
    def map(self, tau):
        return np.random.uniform(self.xlb, self.xub).astype(np.int)


class SingleSolution(LearningBasedMap):
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, verbose=0, data_path=None, data_size=None, **kwargs):
        super().__init__(scene_name=scene_name, case=case, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)
        self.sol = kwargs['sol']
        self.map_name = 'ss'

    def load_sol(self, sol):
        self.sol = sol

    def map(self, tau):
        return np.clip(self.sol, self.xlb, self.xub).astype(np.int)


class MixtureModel(LearningBasedMap):
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, verbose=0, data_path=None, data_size=None, **kwargs):
        super().__init__(scene_name=scene_name, case=case, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)
        self.map_name = 'mm'
        self.n_groups = kwargs['sol']
        self.prepare(kwargs['latent_mdl'], kwargs['sub_mdl'])

    def prepare(self, latent_mdl, sub_mdl):
        self.latent_mdl = latent_mdl
        self.sub_mdl = sub_mdl

    def map(self, tau):
        return self.sub_mdl[self.latent_mdl.predict(np.array([tau]))[0]](tau)


class LMM(LearningBasedMap):
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, verbose=0, data_path=None, data_size=None, **kwargs):
        super().__init__(scene_name=scene_name, case=case, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)
        with open(os.path.join(config.PROJECT_PATH, "data", "mdl", self.scene_name, f"c{self.case}",
                               f"lmm_r{self.coll_run_id}.pkl"), 'rb') as f:
            self.mdl = pickle.load(f)

    def map(self, tau):
        return np.clip(self.mdl.map(tau), self.xlb, self.xub)


class LMM_c1(LMM):
    def __init__(self, scene_name, case, coll_algo_name, coll_run_id=0, verbose=0, data_path=None, data_size=None,
                 **kwargs):
        super().__init__(scene_name=scene_name, case=1, coll_algo_name=coll_algo_name, coll_run_id=coll_run_id,
                         verbose=verbose, data_path=data_path, data_size=data_size)


class OptimizationBasedMap:
    def __init__(self, opt_algo_name, scene_name, case, inst_id,
                 n_workers=1, run_id=0, save=True, maxnfe=1000, simulation_id=None):
        self.opt_algo_name = opt_algo_name
        self.scene_name = scene_name
        self.case = case
        self.inst_id = inst_id
        self.cfg = config.config_dictionary[self.scene_name]
        self.pg = traffic_problem.TSTOProblemGenerator(scene_name, case, simulation_id=simulation_id)
        self.ev = traffic_problem.TrafficPerformanceEvaluator('spmi', n_workers=n_workers)
        self.xlb = self.cfg.get_xlb()
        self.xub = self.cfg.get_xub()
        self.run_id = run_id
        self.save = save
        self.maxnfe = maxnfe

        if self.save:
            self.algo_data_dir = os.path.join(config.PROJECT_PATH, "data", "test",
                                              self.scene_name, f"c{self.case}", self.opt_algo_name)
            if not os.path.exists(self.algo_data_dir):
                os.makedirs(self.algo_data_dir)
            self.algo_data_path = os.path.join(self.algo_data_dir,
                                               'i' + str(self.inst_id) + '_r' + str(self.run_id) + '.npy')


class GA(OptimizationBasedMap):
    def __init__(self, scene_name, case, inst_id, n_workers=1, run_id=0, save=True, maxnfe=1000, simulation_id=None):
        super().__init__('ga', scene_name, case, inst_id,
                         n_workers=n_workers, run_id=run_id, save=save, maxnfe=maxnfe, simulation_id=simulation_id)

    def map(self, tau, pool):
        if pool is None:
            raise Exception("Invalid input process pool")
        prob = self.pg.get_prob_instance(tau)
        ga = optimizer.GA(prob, self.ev, maxnfe=self.maxnfe, rnd_seed=self.run_id)
        ga.initialize(pool=pool)
        ga.iterate(pool=pool, verbose=1)
        X, _ = ga.get_best_data()
        X_all, Y_all = ga.get_all_data()
        np.save(self.algo_data_path, np.c_[X_all, Y_all])
        return np.clip(X, self.xlb, self.xub)


class DE(OptimizationBasedMap):
    def __init__(self, scene_name, case, inst_id, n_workers=1, run_id=0, save=True, maxnfe=1000, simulation_id=None):
        super().__init__('de', scene_name, case, inst_id,
                         n_workers=n_workers, run_id=run_id, save=save, maxnfe=maxnfe, simulation_id=simulation_id)

    def map(self, tau, pool):
        if pool is None:
            raise Exception("Invalid input process pool")
        prob = self.pg.get_prob_instance(tau)
        de = optimizer.DE(prob, self.ev, maxnfe=self.maxnfe, rnd_seed=self.run_id)
        de.initialize(pool=pool)
        de.iterate(pool=pool, verbose=1)
        X, _ = de.get_best_data()
        X_all, Y_all = de.get_all_data()
        np.save(self.algo_data_path, np.c_[X_all, Y_all])
        return np.clip(X, self.xlb, self.xub)


class PSO(OptimizationBasedMap):
    def __init__(self, scene_name, case, inst_id, n_workers=1, run_id=0, save=True, maxnfe=1000, simulation_id=None):
        super().__init__('pso', scene_name, case, inst_id,
                         n_workers=n_workers, run_id=run_id, save=save, maxnfe=maxnfe, simulation_id=simulation_id)

    def map(self, tau, pool):
        if pool is None:
            raise Exception("Invalid input process pool")
        prob = self.pg.get_prob_instance(tau)
        pso = optimizer.PSO(prob, self.ev, maxnfe=self.maxnfe, rnd_seed=self.run_id)
        pso.initialize(pool=pool)
        pso.iterate(pool=pool, verbose=0)
        X, _ = pso.get_best_data()
        X_all, Y_all = pso.get_all_data()
        np.save(self.algo_data_path, np.c_[X_all, Y_all])
        return np.clip(X, self.xlb, self.xub)


class LMM20(OptimizationBasedMap):
    def __init__(self, scene_name, case, inst_id, n_workers=1, run_id=0,
                 save=True, maxnfe=1000, simulation_id=None, coll_run_id=0):
        super().__init__('lmm20', scene_name, case, inst_id,
                         n_workers=n_workers, run_id=run_id, save=save, maxnfe=maxnfe, simulation_id=simulation_id)
        self.coll_run_id = coll_run_id

    def map(self, tau, pool):
        if pool is None:
            raise Exception("Invalid input process pool")
        num_sols = 20

        # transfer (N-1) nearest neighbor solutions based on MTPSO-collected dataset
        data_tau, data_xopt = utils.load_dataset('mtpso', self.scene_name, self.case, run_id=self.coll_run_id)
        tdists = np.array([np.sum(np.abs(tau - tau0)) for tau0 in data_tau])
        sorted_index = np.argsort(tdists)
        sols = data_xopt[sorted_index[:num_sols - 1], :]

        # transfer 1 predicted solution by the LMM-obtained map model
        with open(os.path.join(config.PROJECT_PATH, "data", "mdl", self.scene_name, f"c{self.case}",
                               f"lmm_r{self.coll_run_id}.pkl"), 'rb') as f:
            mdl = pickle.load(f)
        sols = np.r_[sols, mdl.map(tau).reshape(1, -1)]
        prob = self.pg.get_prob_instance(tau)
        fit = self.ev.evaluate(prob, sols, pool=pool)
        xb = sols[np.argmin(fit), :]
        np.save(self.algo_data_path, np.c_[sols, fit])
        return np.clip(xb, self.xlb, self.xub)


class LMMPSO(OptimizationBasedMap):
    def __init__(self, scene_name, case, inst_id, n_workers=1, run_id=0,
                 save=True, maxnfe=1000, simulation_id=None, coll_run_id=0):
        super().__init__('lmmpso', scene_name, case, inst_id,
                         n_workers=n_workers, run_id=run_id, save=save, maxnfe=maxnfe, simulation_id=simulation_id)
        self.coll_run_id = coll_run_id

    def map(self, tau, pool):
        if pool is None:
            raise Exception("Invalid input process pool")
        num_trsols = config.optimizer_hp['TrPSO']['num_trsols']

        # transfer (N-1) nearest neighbor solutions based on MTPSO-collected dataset
        data_tau, data_xopt = utils.load_dataset('mtpso', self.scene_name, self.case, run_id=self.coll_run_id)
        tdists = np.array([np.sum(np.abs(tau - tau0)) for tau0 in data_tau])
        sorted_index = np.argsort(tdists)
        trsols = data_xopt[sorted_index[:num_trsols - 1], :]

        # transfer 1 predicted solution by the LMM-obtained map model
        with open(os.path.join(config.PROJECT_PATH, "data", "mdl", self.scene_name, f"c{self.case}",
                               f"lmm_r{self.coll_run_id}.pkl"), 'rb') as f:
            mdl = pickle.load(f)
        trsols = np.r_[trsols, mdl.map(tau).reshape(1,-1)]

        prob = self.pg.get_prob_instance(tau)
        trpso = traffic_optimizer.TrPSO(prob, self.ev, maxnfe=config.search_fes[self.scene_name],
                                        maxstag=config.maxstag,
                                        tau=tau,
                                        rnd_seed=self.run_id)
        trpso.initialize_with_trsols(trsols=trsols, pool=pool)
        nfe = trpso.iterate(pool=pool, verbose=0)
        xb, _ = trpso.get_best_data()
        X_all, Y_all = trpso.get_all_data()
        print("|", ("nfe=" + str(nfe)).ljust(config.display_bar_len), "|")
        np.save(self.algo_data_path, np.c_[X_all, Y_all])
        return np.clip(xb, self.xlb, self.xub)


class CLPSO(OptimizationBasedMap):
    def __init__(self, scene_name, case, inst_id, n_workers=1, run_id=0, save=True, maxnfe=1000, simulation_id=None):
        super().__init__('clpso', scene_name, case, inst_id,
                         n_workers=n_workers, run_id=run_id, save=save,
                         maxnfe=maxnfe, simulation_id=simulation_id)

    def map(self, tau, pool):
        if pool is None:
            raise Exception("Invalid input process pool")
        prob = self.pg.get_prob_instance(tau)
        clpso = optimizer.CLPSO(prob, self.ev, maxnfe=self.maxnfe, rnd_seed=self.run_id)
        clpso.initialize(pool=pool)
        nfe = clpso.iterate(pool=pool, verbose=0)
        X, _ = clpso.get_best_data()
        X_all, Y_all = clpso.get_all_data()
        print("|", ("nfe=" + str(nfe)).ljust(config.display_bar_len), "|")
        np.save(self.algo_data_path, np.c_[X_all, Y_all])
        return np.clip(X, self.xlb, self.xub)


class MELPSO(OptimizationBasedMap):
    '''
        Implementation of "Knowledge Embedding-Assisted Multi-Exemplar Learning Particle Swarm Optimization for Traffic
        Signal Timing Optimization".
    '''
    def __init__(self, scene_name, case, inst_id, n_workers=1, run_id=0, save=True, maxnfe=1000, simulation_id=None):
        super().__init__('melpso', scene_name, case, inst_id, n_workers=n_workers, run_id=run_id, save=save,
                         maxnfe=maxnfe, simulation_id=simulation_id)

    def map(self, tau, pool):
        if pool is None:
            raise Exception("Invalid input process pool")
        prob = self.pg.get_prob_instance(tau)
        melpso = traffic_optimizer.MELPSO(prob, self.ev, maxnfe=self.maxnfe, rnd_seed=self.run_id)
        melpso.initialize(pool=pool)
        nfe = melpso.iterate(pool=pool, verbose=0)
        X, _ = melpso.get_best_data()
        X_all, Y_all = melpso.get_all_data()
        print("|", ("nfe=" + str(nfe)).ljust(config.display_bar_len), "|")
        np.save(self.algo_data_path, np.c_[X_all, Y_all])
        return np.clip(X, self.xlb, self.xub)


class REPSO(OptimizationBasedMap):
    '''
        Implementation of "Region-based Evaluation Particle Swarm Optimization with Dual Solution Libraries for
        Real-time Traffic Signal Timing Optimization".
    '''
    def __init__(self, scene_name, case, inst_id, n_workers=1, run_id=0,
                 save=True, maxnfe=1000, simulation_id=None, coll_run_id=0):
        super().__init__('repso', scene_name, case, inst_id, n_workers=n_workers, run_id=run_id, save=save,
                         maxnfe=maxnfe, simulation_id=simulation_id)
        self.coll_run_id = coll_run_id

    def map(self, tau, pool):
        if pool is None:
            raise Exception("Invalid input process pool")
        num_trsols = config.optimizer_hp['REPSO']['num_trsols']
        data_tau, data_xopt = utils.load_dataset('spso', self.scene_name, self.case, run_id=self.coll_run_id)
        tdists = np.array([np.linalg.norm(tau - tau0) for tau0 in data_tau])
        sorted_index = np.argsort(tdists)
        trsols = data_xopt[sorted_index[:num_trsols], :]
        prob = self.pg.get_prob_instance(tau)
        repso = traffic_optimizer.REPSO(prob, self.ev, maxnfe=config.search_fes[self.scene_name],
                                        tau=tau, rnd_seed=self.run_id)
        repso.initialize(trsols=trsols, pool=pool)
        nfe = repso.iterate(pool=pool, verbose=0)
        xb, _ = repso.get_best_data()
        X_all, Y_all = repso.get_all_data()
        print("|", ("nfe=" + str(nfe)).ljust(config.display_bar_len), "|")
        np.save(self.algo_data_path, np.c_[X_all, Y_all])
        return np.clip(xb, self.xlb, self.xub)


def get_optimization_map(optimizer_name):
    if optimizer_name == 'pso':
        return PSO
    elif optimizer_name == 'lmm20':
        return LMM20
    elif optimizer_name == 'lmmpso':
        return LMMPSO
    elif optimizer_name == 'clpso':
        return CLPSO
    elif optimizer_name == 'repso':
        return REPSO
    elif optimizer_name == 'melpso':
        return MELPSO
    elif optimizer_name == 'de':
        return DE
    elif optimizer_name == 'ga':
        return GA
    else:
        raise ValueError


def get_learning_map(map_name):
    if map_name == 'lmm':
        return LMM
    elif map_name == 'lmm_c1':
        return LMM_c1
    elif map_name == 'lr':
        return LinearMap
    elif map_name == 'nn':
        return NeuralMap
    elif map_name == 'dt':
        return DecisionTreeMap
    elif map_name == 'rf':
        return RandomForestMap
    elif map_name == 'gs':
        return GaussianSample
    elif map_name == 'mos':
        return MeanOfSample
    elif map_name == 'rs':
        return RandomSelection
    elif map_name == 'knn':
        return NearestNeighborOffline
    elif map_name == 'us':
        return UniformSample
    elif map_name == 'ss':
        return SingleSolution
    elif map_name == 'mm':
        return MixtureModel
    else:
        raise ValueError


if __name__ == "__main__":
    pass


