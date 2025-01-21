# coding: utf-8
import numpy as np
import copy
from copy import deepcopy
import transfer
from sklearn.cluster import KMeans


class Grid:
    def __init__(self, prob, eval):
        self.prob = prob
        self.eval = eval
        self.data_X = []
        self.data_Y = []
        if self.prob.get_dim_sol() > 2:
            raise Exception("The problem dimensionality is too large for enumeration")

    def sample(self, pool=None):
        x0 = np.arange(self.prob.get_xlb()[0], self.prob.get_xub()[0])
        x1 = np.arange(self.prob.get_xlb()[1], self.prob.get_xub()[1])
        n0 = len(x0)
        n1 = len(x1)
        nfe = 0

        while nfe < n0 * n1:
            sols = []
            for i in range(self.eval.batch_size):
                i0, i1 = nfe // n0, nfe % n1
                sols.append(np.array([x0[i0], x1[i1]]))
                nfe += 1
                if nfe >= n0 * n1:
                    break

            sols = np.array(sols)
            fits = self.eval.evaluate(self.prob, sols, pool=pool)
            for i in range(sols.shape[0]):
                self.data_X.append(copy.deepcopy(sols[i]))
                self.data_Y.append(fits[i])
        return nfe

    def get_all_data(self):
        return np.array(self.data_X), np.array(self.data_Y)


class TrPSO:
    def __init__(self, prob, eval, maxnfe, tau=None, maxstag=1e25, rnd_seed=0):
        if tau is None:
            tau = []
        if rnd_seed:
            np.random.seed(rnd_seed)
        self.prob = prob
        self.eval = eval
        self.data_X = []
        self.data_Y = []
        self.ps = 20
        self.xlb = self.prob.get_xlb()
        self.xub = self.prob.get_xub()
        self.vlb = -0.1 * (self.xub - self.xlb)
        self.vub = 0.1 * (self.xub - self.xlb)
        self.dim = self.prob.get_dim_sol()
        if self.dim == 1:
            self.ps = 5
        self.nfe = 0
        self.stag_nfe = 0
        self.maxnfe = maxnfe
        self.maxstag = maxstag
        self.tau = tau

    def initialize_with_trsols(self, trsols, pool=None):
        if trsols is not None:
            merged_pos = np.r_[trsols, np.random.uniform(self.xlb, self.xub, (self.ps, self.dim))]
        else:
            merged_pos = np.random.uniform(self.xlb, self.xub, (self.ps, self.dim))
        merged_fit = self.eval.evaluate(self.prob, merged_pos, pool=pool)
        sorted_index = np.argsort(merged_fit)
        self.pos = merged_pos[sorted_index[:self.ps], :]
        self.vel = np.random.uniform(self.vlb, self.vub, (self.ps, self.dim))
        self.fit = merged_fit[sorted_index[:self.ps]]
        self.nfe += merged_pos.shape[0]
        self.pbx = deepcopy(self.pos)
        self.pby = deepcopy(self.fit)
        self.gbx = deepcopy(self.pbx[np.argmin(self.pby), :])
        self.gby = np.min(self.pby)
        for i in range(merged_pos.shape[0]):
            self.data_X.append(deepcopy(merged_pos[i,:]))
            self.data_Y.append(merged_fit[i])

    def initialize_with_trfunc(self, trfunc, xlb=None, xub=None, pool=None):
        if len(self.tau) == 0:
            if xlb is not None and xub is not None:
                if np.sum(xlb > xub) > 0:
                    raise Exception("The lower bound shoud be smaller than or equal to upper bound.")
                reg_xlb, reg_xub = np.clip(xlb, self.xlb, self.xub), np.clip(xub, self.xlb, self.xub)
                vlb, vub = -0.1 * (reg_xub - reg_xlb), 0.1 * (reg_xub - reg_xlb)
                self.pos = np.random.uniform(reg_xlb, reg_xub, (self.ps, self.dim))
                self.vel = np.random.uniform(vlb, vub, (self.ps, self.dim))
            else:
                self.pos = np.random.uniform(self.xlb, self.xub, (self.ps, self.dim))
                self.vel = np.random.uniform(self.vlb, self.vub, (self.ps, self.dim))
            self.fit = self.eval.evaluate(self.prob, self.pos, pool=pool)
        else:
            if xlb is not None and xub is not None:
                if np.sum(xlb > xub) > 0:
                    raise Exception("The lower bound shoud be smaller than or equal to upper bound.")
                reg_xlb, reg_xub = np.clip(xlb, self.xlb, self.xub), np.clip(xub, self.xlb, self.xub)
                vlb, vub = -0.1 * (reg_xub - reg_xlb), 0.1 * (reg_xub - reg_xlb)
                self.pos = np.random.uniform(reg_xlb, reg_xub, (self.ps, self.dim))
                self.vel = np.random.uniform(vlb, vub, (self.ps, self.dim))
            else:
                self.pos = np.random.uniform(self.xlb, self.xub, (self.ps, self.dim))
                self.vel = np.random.uniform(self.vlb, self.vub, (self.ps, self.dim))
            sol = trfunc(self.tau)
            self.pos = np.random.uniform(self.xlb, self.xub, (self.ps, self.dim))
            self.pos[-1, :] = sol
            self.fit = self.eval.evaluate(self.prob, self.pos, pool=pool)
            # print("transferred x:", sol, "y:", np.round(self.fit[-1],3))
        self.nfe += self.ps
        self.pbx = deepcopy(self.pos)
        self.pby = deepcopy(self.fit)
        self.gbx = deepcopy(self.pbx[np.argmin(self.pby), :])
        self.gby = np.min(self.pby)
        for i in range(self.pos.shape[0]):
            self.data_X.append(deepcopy(self.pos[i, :]))
            self.data_Y.append(self.fit[i])

    def iterate_with_mst(self, mst, tr_params=None, pool=None, verbose=0):
        n_trsols = tr_params['num_trsols']
        iteration = 0
        while self.nfe < self.maxnfe and self.stag_nfe < self.maxstag:
            if verbose == 1:
                print("nfe", self.nfe, "gbfit", self.gby)
            w = 0.9 - 0.5 * self.nfe / self.maxnfe
            for i in range(self.ps):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.vel[i, :] = w * self.vel[i, :] + 2.0 * r1 * (self.pbx[i, :] - self.pos[i, :]) + 2.0 * r2 * (
                            self.gbx - self.pos[i, :])
                self.vel[i, :] = np.clip(self.vel[i, :], self.vlb, self.vub)
                self.pos[i, :] = np.clip(self.pos[i, :] + self.vel[i, :], self.xlb, self.xub)

            self.fit = self.eval.evaluate(self.prob, self.pos, pool=pool)
            self.nfe += self.ps
            if iteration > 0 and iteration % 5 == 0:
                # periodically perform knowledge transfer from the collected knowledge base
                trsols = mst.ask('posterior', self.prob.net_input, n_trsols, state={'xb': self.gbx})
                if trsols is not None:
                    trfit = self.eval.evaluate(self.prob, trsols, pool=pool)
                    print('transfer at iter', iteration, 'best trfit', np.min(trfit))
                    if np.min(trfit) < self.gby:
                        self.gby = np.min(trfit)
                        self.gbx = deepcopy(trsols[np.argmin(trfit), :])

            for i in range(self.ps):
                self.data_X.append(deepcopy(self.pos[i, :]))
                self.data_Y.append(self.fit[i])
                if self.fit[i] < self.pby[i]:
                    self.pby[i] = self.fit[i]
                    self.pbx[i, :] = deepcopy(self.pos[i, :])
                if self.fit[i] < self.gby:
                    self.stag_nfe = 0
                    self.gby = self.fit[i]
                    self.gbx = deepcopy(self.pos[i, :])
                else:
                    self.stag_nfe += 1

                if self.stag_nfe >= self.maxstag:
                    break

            iteration += 1

        return self.nfe

    def iterate(self, pool=None, verbose=0):
        while self.nfe < self.maxnfe and self.stag_nfe < self.maxstag:
            if verbose == 1:
                print("nfe", self.nfe, "gbfit", self.gby)
            w = 0.9 - 0.5 * self.nfe / self.maxnfe
            for i in range(self.ps):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.vel[i, :] = w * self.vel[i, :] + 2.0 * r1 * (self.pbx[i, :] - self.pos[i, :]) + 2.0 * r2 * (
                            self.gbx - self.pos[i, :])
                self.vel[i, :] = np.clip(self.vel[i, :], self.vlb, self.vub)
                self.pos[i, :] = np.clip(self.pos[i, :] + self.vel[i, :], self.xlb, self.xub)

            self.fit = self.eval.evaluate(self.prob, self.pos, pool=pool)
            self.nfe += self.ps

            for i in range(self.ps):
                self.data_X.append(deepcopy(self.pos[i, :]))
                self.data_Y.append(self.fit[i])
                if self.fit[i] < self.pby[i]:
                    self.pby[i] = self.fit[i]
                    self.pbx[i, :] = deepcopy(self.pos[i, :])
                if self.fit[i] < self.gby:
                    self.stag_nfe = 0
                    self.gby = self.fit[i]
                    self.gbx = deepcopy(self.pos[i, :])
                else:
                    self.stag_nfe += 1

                if self.stag_nfe >= self.maxstag:
                    break
        return self.nfe

    def get_all_data(self):
        return np.array(self.data_X), np.array(self.data_Y)

    def get_best_data(self):
        X, Y = self.get_all_data()
        return X[np.argmin(Y), :], np.min(Y)

    def get_batch_data(self, batch_size):
        X, Y = self.get_all_data()
        return X[:batch_size, :], Y[:batch_size]

    def get_sliced_best_data(self, slices):
        slices = np.sort(np.array(slices))
        X, Y = self.get_all_data()
        retX, retY = [], []
        for i in range(len(slices)):
            best_index = np.argmin(Y[:slices[i] + 1])
            retX.append(X[best_index, :])
            retY.append(Y[best_index])
        retX, retY = np.array(retX), np.array(retY)
        return retX, retY


class MELPSO:
    '''
        Implementation of "Knowledge Embedding-Assisted Multi-Exemplar Learning Particle Swarm Optimization for Traffic
        Signal Timing Optimization".
    '''
    def __init__(self, prob, eval, maxnfe, maxstag=1e25, rnd_seed=0):
        if rnd_seed:
            np.random.seed(rnd_seed)
        self.prob = prob
        self.eval = eval
        self.data_X = []
        self.data_Y = []
        self.ps = 20
        self.xlb = self.prob.xlb
        self.xub = self.prob.xub
        self.vlb = -0.1 * (self.xub - self.xlb)
        self.vub = 0.1 * (self.xub - self.xlb)
        self.dim = self.prob.dim
        if self.dim < 2:
            self.ps = 5
        elif self.dim >= 100:
            self.ps = 50
        self.nfe = 0
        self.stag_nfe = 0
        self.maxnfe = maxnfe
        self.maxstag = maxstag
        self.kt = 0.8

    def initialize(self, pool=None):
        tau = self.prob.net_input
        # KESG - Knowledge Embedded Solution Generating
        web_trobj = transfer.WebsterTransfer(self.prob.scene_name, self.prob.case)
        web_trobj.load_webster()
        webster_x = web_trobj.map(tau)
        pos = []
        for i in range(self.ps):
            if np.random.random() < self.kt:
                pos.append(np.clip(webster_x + np.random.normal(loc=[0] * self.dim, scale=[5] * self.dim), self.xlb, self.xub))
            else:
                pos.append(np.random.uniform(self.xlb, self.xub))
        self.pos = np.array(pos)
        self.vel = np.random.uniform(self.vlb, self.vub, (self.ps, self.dim))
        self.fit = self.eval.evaluate(self.prob, self.pos, pool=pool)
        self.nfe += self.ps
        self.pbx = deepcopy(self.pos)
        self.pby = deepcopy(self.fit)
        self.gbx = deepcopy(self.pbx[np.argmin(self.pby), :])
        self.gby = np.min(self.pby)
        for i in range(self.pos.shape[0]):
            self.data_X.append(deepcopy(self.pos[i, :]))
            self.data_Y.append(self.fit[i])

        self.w0 = 0.9  # upper bound of inertia weight
        self.w1 = 0.4  # lower bound of inertia weight
        self.m = 7     # refreshing gap
        self.c1 = 0.75  # acceleration coefficient 1
        self.c2 = 0.75  # acceleration coefficient 2
        self.c3 = 1.5   # acceleration coefficient 3

        self.flag = np.zeros(self.ps)
        self.fi = np.array([np.ones(self.dim) * i for i in range(self.ps)]).astype(np.int)

        for i in range(self.ps):
            Pc = 0.05 + 0.45 * (np.exp(10 * i / (self.ps - 1)) - 1) / (np.exp(10) - 1)
            for d in range(self.dim):
                if np.random.random() < Pc:
                    f1 = np.random.randint(self.ps)
                    f2 = np.random.randint(self.ps)
                    while f1 == i:
                        f1 = np.random.randint(self.ps)
                    while f2 == i:
                        f2 = np.random.randint(self.ps)
                    if self.pby[f1] < self.pby[f2]:
                        self.fi[i, d] = f1
                    else:
                        self.fi[i, d] = f2
                else:
                    self.fi[i, d] = i

    def iterate(self, pool=None, verbose=0):
        iteration = 0
        maxiter = int(self.maxnfe / self.ps)
        while self.nfe < self.maxnfe and self.stag_nfe < self.maxstag:
            if verbose == 1:
                print("nfe", self.nfe, "gbfit", self.gby)
            w = self.w0 - (self.w1 - self.w0) * iteration / maxiter
            for i in range(self.ps):
                if self.flag[i] >= self.m:
                    self.flag[i] = 0
                    Pc = 0.05 + 0.45 * (np.exp(10 * i / (self.ps - 1)) - 1) / (np.exp(10) - 1)
                    for d in range(self.dim):
                        if np.random.random() < Pc:
                            f1 = np.random.randint(self.ps)
                            f2 = np.random.randint(self.ps)
                            while f1 == i:
                                f1 = np.random.randint(self.ps)
                            while f2 == i:
                                f2 = np.random.randint(self.ps)
                            if self.pby[f1] < self.pby[f2]:
                                self.fi[i, d] = f1
                            else:
                                self.fi[i, d] = f2
                        else:
                            self.fi[i, d] = i
                for d in range(self.dim):
                    self.vel[i, d] = w * self.vel[i, d] + self.c1 * np.random.random() * (self.pbx[i, d] - self.pos[i, d])\
                                     + self.c2 * np.random.random() * (self.gbx[d] - self.pos[i, d])\
                                     + self.c3 * np.random.random() * (self.pbx[self.fi[i, d], d] - self.pos[i, d])
                    self.vel[i, d] = np.clip(self.vel[i, d], self.vlb[d], self.vub[d])
                    self.pos[i, d] = np.clip(self.pos[i, d] + self.vel[i, d], self.xlb[d], self.xub[d])
            self.fit = self.eval.evaluate(self.prob, self.pos, pool=pool)
            self.nfe += self.ps
            for i in range(self.ps):
                self.data_X.append(deepcopy(self.pos[i, :]))
                self.data_Y.append(self.fit[i])
                if self.fit[i] < self.pby[i]:
                    self.pby[i] = self.fit[i]
                    self.pbx[i, :] = deepcopy(self.pos[i, :])
                else:
                    self.flag[i] += 1
                if self.fit[i] < self.gby:
                    self.stag_nfe = 0
                    self.gby = self.fit[i]
                    self.gbx = deepcopy(self.pos[i, :])
                else:
                    self.stag_nfe += 1

                if self.stag_nfe >= self.maxstag:
                    break
        return self.nfe

    def get_all_data(self):
        return np.array(self.data_X), np.array(self.data_Y)

    def get_best_data(self):
        X, Y = self.get_all_data()
        return X[np.argmin(Y), :], np.min(Y)

    def get_batch_data(self, batch_size):
        X, Y = self.get_all_data()
        return X[:batch_size, :], Y[:batch_size]

    def get_sliced_best_data(self, slices):
        slices = np.sort(np.array(slices))
        X, Y = self.get_all_data()
        retX, retY = [], []
        for i in range(len(slices)):
            best_index = np.argmin(Y[:slices[i] + 1])
            retX.append(X[best_index, :])
            retY.append(Y[best_index])
        retX, retY = np.array(retX), np.array(retY)
        return retX, retY


class REPSO:
    '''
        Implementation of "Region-based Evaluation Particle Swarm Optimization with Dual Solution Libraries for
        Real-time Traffic Signal Timing Optimization".
    '''
    def __init__(self, prob, eval, maxnfe, tau=None, rnd_seed=0):
        if tau is None:
            tau = []
        if rnd_seed:
            np.random.seed(rnd_seed)
        self.prob = prob
        self.eval = eval
        self.data_X = []
        self.data_Y = []
        self.ps = 100
        self.K = 20     # number of regions in RES
        self.TP = 0.05  # probability of KIS
        self.xlb = self.prob.get_xlb()
        self.xub = self.prob.get_xub()
        self.vlb = -0.1 * (self.xub - self.xlb)
        self.vub = 0.1 * (self.xub - self.xlb)
        self.dim = self.prob.get_dim_sol()
        self.nfe = 0
        self.maxnfe = maxnfe
        self.tau = tau

    def cluster(self, pos):
        km = KMeans(n_clusters=self.K, random_state=1).fit(pos)
        labels_ = km.labels_
        ctr_pos = []
        for k in range(self.K):
            ctr_pos.append(np.mean(pos[labels_ == k,:],axis=0))
        ctr_pos = np.array(ctr_pos)
        return labels_, ctr_pos

    def initialize(self, trsols, pool=None):
        # KIS
        if trsols is not None and np.random.random() < self.TP:
            self.pos = np.r_[trsols, np.random.uniform(self.xlb, self.xub, (self.ps - trsols.shape[0], self.dim))]
        else:
            self.pos = np.random.uniform(self.xlb, self.xub, (self.ps, self.dim))
        self.fit = np.ones(self.pos.shape[0]) * 1e25
        # RES
        self.region_labels, self.ctr_pos = self.cluster(self.pos)
        self.ctr_fit = self.eval.evaluate(self.prob, self.ctr_pos, pool=pool)
        for k in range(self.K):
            self.fit[self.region_labels == k] = self.ctr_fit[k]
        self.nfe += self.ctr_pos.shape[0]

        self.vel = np.random.uniform(self.vlb, self.vub, (self.ps, self.dim))
        self.pbx = deepcopy(self.pos)
        self.pby = deepcopy(self.fit)
        self.gbx = deepcopy(self.pbx[np.argmin(self.pby), :])
        self.gby = np.min(self.pby)
        for i in range(self.ctr_pos.shape[0]):
            self.data_X.append(deepcopy(self.ctr_pos[i, :]))
            self.data_Y.append(self.ctr_fit[i])

    def iterate(self, pool=None, verbose=0):
        while self.nfe < self.maxnfe:
            if verbose == 1:
                print("nfe", self.nfe, "gbfit", self.gby)
            # sort the regions to get rank
            sort_index = np.argsort(-self.ctr_fit)
            region_ranks = np.argsort(sort_index) # the larger, the better
            label_to_pos = []
            for k in range(self.K):
                label_to_pos.append(self.pos[self.region_labels == k, :])

            for i in range(self.ps):
                region_rank = region_ranks[self.region_labels[i]]
                for d in range(self.dim):
                    L1 = np.random.randint(self.K - region_rank) + region_rank
                    L2 = np.random.randint(self.K - region_rank) + region_rank
                    p1 = np.random.randint(label_to_pos[L1].shape[0])
                    p2 = np.random.randint(label_to_pos[L2].shape[0])
                    r1 = np.random.random()
                    r2 = np.random.random()
                    r3 = np.random.random()
                    self.vel[i, d] = r1 * self.vel[i, d] + r2 * (label_to_pos[L1][p1, d] - self.pos[i, d]) + \
                                     L2 / self.K * r3 * (label_to_pos[L2][p2, d] - self.pos[i, d])
                    self.vel[i, d] = np.clip(self.vel[i, d], self.vlb[d], self.vub[d])
                    self.pos[i, d] = np.clip(self.pos[i, d] + self.vel[i, d], self.xlb[d], self.xub[d])

            # RES
            self.region_labels, self.ctr_pos = self.cluster(self.pos)
            self.ctr_fit = self.eval.evaluate(self.prob, self.ctr_pos, pool=pool)
            for k in range(self.K):
                self.fit[self.region_labels == k] = self.ctr_fit[k]
            self.nfe += self.ctr_pos.shape[0]

            for i in range(self.ctr_pos.shape[0]):
                self.data_X.append(deepcopy(self.ctr_pos[i, :]))
                self.data_Y.append(self.ctr_fit[i])

            for i in range(self.ps):
                if self.fit[i] < self.pby[i]:
                    self.pby[i] = self.fit[i]
                    self.pbx[i, :] = deepcopy(self.pos[i, :])
                if self.fit[i] < self.gby:
                    self.gby = self.fit[i]
                    self.gbx = deepcopy(self.pos[i, :])

        return self.nfe

    def get_all_data(self):
        return np.array(self.data_X), np.array(self.data_Y)

    def get_best_data(self):
        X, Y = self.get_all_data()
        return X[np.argmin(Y), :], np.min(Y)


class BGA_ML:
    '''
        Implementation of "Boosted genetic algorithm using machine learning for traffic control optimization".
    '''
    def __init__(self, prob, eval, maxnfe, tau=None, maxstag=1e25, rnd_seed=0):
        if tau is None:
            tau = []
        if rnd_seed:
            np.random.seed(rnd_seed)
        self.prob = prob
        self.eval = eval
        self.data_X = []
        self.data_Y = []
        self.ps = 50
        self.nfe = 0
        self.maxnfe = maxnfe
        self.tau = tau
        self.maxstag = maxstag
        self.pop = []
        self.fit = []

    def initialize(self, pool=None):
        self.pop = np.array(self.prob.sample_x(nx=self.ps))
        self.fit = self.eval.evaluate(self.prob, self.pop, pool=pool)
        self.nfe += self.ps
        for i in range(self.ps):
            self.data_X.append(deepcopy(self.pop[i, :]))
            self.data_Y.append(self.fit[i])
        self.gbx = deepcopy(self.pop[np.argmin(self.fit), :])
        self.gby = np.min(self.fit)
        self.stag_nfe = 0

    def crossover(self, p1, p2, p=0.8):
        if np.random.rand() <= p:
            lin_c1 = np.random.rand()
            lin_c2 = np.random.rand()
            child1 = p1 * lin_c1 + p2 * (1 - lin_c1)
            child2 = p1 * lin_c2 + p2 * (1 - lin_c2)
            return child1, child2
        else:
            return p1, p2

    def mutation(self, individual, p=0.1):
        if np.random.rand() <= p:
            mutant = deepcopy(individual)
            tls_id = np.random.randint(len(self.prob.cfg.ts_ids))
            start_dim = int(np.sum(self.prob.cfg.n_phases[:tls_id]))
            phase_ids = np.arange(start_dim, start_dim + self.prob.cfg.n_phases[tls_id])
            dim1, dim2 = phase_ids[np.random.permutation(len(phase_ids))[:2]]
            duration1, duration2 = mutant[dim1], mutant[dim2]
            if np.random.rand() <= 0.5:
                duration_var = np.random.rand() * (duration1 - self.prob.min_duration)
                mutant[dim1] -= duration_var
                mutant[dim2] += duration_var
            else:
                duration_var = np.random.rand() * (duration2 - self.prob.min_duration)
                mutant[dim1] += duration_var
                mutant[dim2] -= duration_var
            return mutant
        else:
            return individual

    def iterate(self, pool=None, verbose=0):
        while self.nfe < self.maxnfe and self.stag_nfe < self.maxstag:
            if verbose == 1:
                print("nfe", self.nfe, "gbfit", self.gby)

            # Selection - Tournament Selection
            new_pop = [deepcopy(self.pop[np.argmin([self.fit[np.random.randint(0, self.ps)] for _ in range(2)])]) for _
                       in range(self.ps)]

            # Crossover - Simulated Binary Crossover
            for i in range(0, self.ps, 2):
                if i + 1 < self.ps:
                    new_pop[i], new_pop[i + 1] = self.crossover(new_pop[i], new_pop[i + 1])

            # Mutation - Polynomial Mutation
            new_pop = [self.mutation(ind) for ind in new_pop]

            # Evaluate new population
            new_pop = np.array(new_pop)
            new_fit = self.eval.evaluate(self.prob, new_pop, pool=pool)
            self.nfe += self.ps

            # Replacement - Elitism
            combined_pop = np.vstack((self.pop, new_pop))
            combined_fit = np.hstack((self.fit, new_fit))
            indices = np.argsort(combined_fit)[:self.ps]
            self.pop = deepcopy(combined_pop[indices])
            self.fit = deepcopy(combined_fit[indices])

            for i in range(self.ps):
                self.data_X.append(deepcopy(new_pop[i, :]))
                self.data_Y.append(new_fit[i])
                if new_fit[i] < self.gby:
                    self.gby = new_fit[i]
                    self.gbx = deepcopy(new_pop[i, :])
                    self.stag_nfe = 0
                else:
                    self.stag_nfe += 1

                if self.stag_nfe >= self.maxstag:
                    break

    def get_all_data(self):
        return np.array(self.data_X), np.array(self.data_Y)

    def get_best_data(self):
        X, Y = self.get_all_data()
        return X[np.argmin(Y), :], np.min(Y)


class PSO_QL:
    '''
        Implementation of "Scheduling Eight-Phase Urban Traffic Light Problems via Ensemble Meta-Heuristics and
        Q-Learning Based Local Search".
    '''
    def __init__(self, prob, eval, maxnfe, tau=None, maxstag=1e25, rnd_seed=0):
        if tau is None:
            tau = []
        if rnd_seed:
            np.random.seed(rnd_seed)
        self.prob = prob
        self.eval = eval
        self.data_X = []
        self.data_Y = []
        self.ps = 20
        self.xlb = self.prob.get_xlb()
        self.xub = self.prob.get_xub()
        self.vlb = -0.1 * (self.xub - self.xlb)
        self.vub = 0.1 * (self.xub - self.xlb)
        self.dim = self.prob.get_dim_sol()
        self.nfe = 0
        self.maxnfe = maxnfe
        self.tau = tau
        self.maxstag = maxstag
        self.action_space = [f"FBLS{i}" for i in range(1, 6)]

    def initialize(self, pool=None):
        self.pos = np.random.uniform(self.xlb, self.xub, (self.ps, self.dim))
        self.fit = self.eval.evaluate(self.prob, self.pos, pool=pool)
        self.nfe += self.pos.shape[0]
        self.vel = np.random.uniform(self.vlb, self.vub, (self.ps, self.dim))
        self.pbx = deepcopy(self.pos)
        self.pby = deepcopy(self.fit)
        self.gbx = deepcopy(self.pbx[np.argmin(self.pby), :])
        self.gby = np.min(self.pby)
        self.stag_nfe = 0
        for i in range(self.pos.shape[0]):
            self.data_X.append(deepcopy(self.pos[i, :]))
            self.data_Y.append(self.fit[i])
        self.q_tbl = np.ones((self.ps, len(self.action_space))) * 1e-10  # 5 stands for five local perturbation schemes

    def perturb(self, param, scheme):
        assert len(param) == self.prob.n_param, f"Expect # param = {self.prob.n_param}, but encounter with {len(param)}"
        solution = np.array(param).reshape(self.prob.n_step, self.prob.n_ts).astype(np.int_)
        if scheme == 'FBLS1':
            signal_id = np.random.randint(self.prob.n_ts)
            for i in range(self.prob.n_step):
                new_phase = np.random.randint(self.prob.full_act_space[signal_id].n)
                while new_phase == solution[i, signal_id]:
                    new_phase = np.random.randint(self.prob.full_act_space[signal_id].n)
                solution[i, signal_id] = new_phase
        elif scheme == 'FBLS2':
            signal_id = np.random.randint(self.prob.n_ts)
            step_id = np.random.randint(self.prob.n_step)
            new_phase = np.random.randint(self.prob.full_act_space[signal_id].n)
            while new_phase == solution[step_id, signal_id]:
                new_phase = np.random.randint(self.prob.full_act_space[signal_id].n)
            solution[step_id, signal_id] = new_phase
        elif scheme == 'FBLS3':
            signal_ids = [np.random.randint(self.prob.n_ts)]
            if len(self.prob.neighbor_ids[signal_ids[0]]) > 0:  # if the signal has a neighbor signal
                signal_ids.append(np.random.choice(self.prob.neighbor_ids[signal_ids[0]]))
            for i in range(self.prob.n_step):
                for signal_id in signal_ids:
                    new_phase = np.random.randint(self.prob.full_act_space[signal_id].n)
                    while new_phase == solution[i, signal_id]:
                        new_phase = np.random.randint(self.prob.full_act_space[signal_id].n)
                    solution[i, signal_id] = new_phase
        elif scheme == 'FBLS4':
            signal_ids = [np.random.randint(self.prob.n_ts)]
            if len(self.prob.neighbor_ids[signal_ids[0]]) > 0:  # if the signal has a neighbor signal
                signal_ids.append(np.random.choice(self.prob.neighbor_ids[signal_ids[0]]))
            step_id = np.random.randint(self.prob.n_step)
            for signal_id in signal_ids:
                new_phase = np.random.randint(self.prob.full_act_space[signal_id].n)
                while new_phase == solution[step_id, signal_id]:
                    new_phase = np.random.randint(self.prob.full_act_space[signal_id].n)
                solution[step_id, signal_id] = new_phase
        elif scheme == 'FBLS5':
            signal_ids = [np.random.randint(self.prob.n_ts)]
            if len(self.prob.non_neighbor_ids[signal_ids[0]]) > 0:  # if the signal has a non-neighbor signal
                signal_ids.append(np.random.choice(self.prob.non_neighbor_ids[signal_ids[0]]))
            step_id = np.random.randint(self.prob.n_step)
            for signal_id in signal_ids:
                new_phase = np.random.randint(self.prob.full_act_space[signal_id].n)
                while new_phase == solution[step_id, signal_id]:
                    new_phase = np.random.randint(self.prob.full_act_space[signal_id].n)
                solution[step_id, signal_id] = new_phase
        else:
            raise ValueError
        return solution.flatten()

    def local_search(self, pool=None):
        sorted_fits = np.sort(self.fit)
        ranks = np.argsort(np.argsort(self.fit))
        alpha = 0.1
        gamma = 0.9
        eps = 0.1
        ls_solution = []
        exp_buffer = []  # experience, (s_t, a_t, r_t, s_{t+1})

        # epsilon-greedy action strategy
        for i in range(self.ps):
            state = ranks[i]
            if np.random.random() < eps:
                action_id = np.random.randint(len(self.action_space))
            else:
                action_id = np.argmax(self.q_tbl[state, :])
            exp_buffer.append([state, action_id])
            ls_solution.append(self.perturb(self.pos[i], scheme=self.action_space[action_id]))
        ls_solution = np.array(ls_solution)
        ls_fitness = self.eval.evaluate(self.prob, ls_solution, pool=pool)
        self.nfe += len(ls_solution)

        # update reward and next state
        for i in range(self.ps):
            exp_buffer[i].append(self.fit[i] - ls_fitness[i])  # reward
            exp_buffer[i].append(np.argwhere(ls_fitness[i] <= np.append(sorted_fits[:-1], 1e25)))  # next state

        # update Q-table
        for exp_tuple in exp_buffer:
            x1, a1, r1, x2 = exp_tuple
            self.q_tbl[x1, a1] = self.q_tbl[x1, a1] + alpha * (r1 + gamma * np.max(self.q_tbl[x2, :]) -
                                                               self.q_tbl[x1, a1])

        # update population
        for i in range(self.ps):
            self.data_X.append(deepcopy(ls_solution[i, :]))
            self.data_Y.append(ls_fitness[i])

            if ls_fitness[i] < self.fit[i]:
                self.fit[i] = ls_fitness[i]
                self.pos[i] = deepcopy(ls_solution[i, :])

                if self.fit[i] < self.pby[i]:
                    self.pby[i] = self.fit[i]
                    self.pbx[i, :] = deepcopy(self.pos[i, :])

                if self.fit[i] < self.gby:
                    self.stag_nfe = 0
                    self.gby = self.fit[i]
                    self.gbx = deepcopy(self.pos[i, :])
                else:
                    self.stag_nfe += 1

                # early stopping
                if self.stag_nfe >= self.maxstag:
                    break

    def iterate(self, pool=None, verbose=0):
        while self.nfe < self.maxnfe and self.stag_nfe < self.maxstag:
            if verbose == 1:
                print("nfe", self.nfe, "gbfit", self.gby)
            w = 0.9 - 0.5 * self.nfe / self.maxnfe
            # if self.gby <= 0.01:
            #     print("gbx", self.gbx)
            #     raise Exception()

            for i in range(self.ps):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.vel[i, :] = w * self.vel[i, :] + 2.0 * r1 * (self.pbx[i, :] - self.pos[i, :]) + 2.0 * r2 * (
                            self.gbx - self.pos[i, :])
                self.vel[i, :] = np.clip(self.vel[i, :], self.vlb, self.vub)
                self.pos[i, :] = np.clip(self.pos[i, :] + self.vel[i, :], self.xlb, self.xub)
            self.fit = self.eval.evaluate(self.prob, self.pos, pool=pool)
            self.nfe += self.ps
            self.local_search(pool=pool)
            for i in range(self.ps):
                self.data_X.append(deepcopy(self.pos[i, :]))
                self.data_Y.append(self.fit[i])
                if self.fit[i] < self.pby[i]:
                    self.pby[i] = self.fit[i]
                    self.pbx[i, :] = deepcopy(self.pos[i, :])
                if self.fit[i] < self.gby:
                    self.stag_nfe = 0
                    self.gby = self.fit[i]
                    self.gbx = deepcopy(self.pos[i, :])
                else:
                    self.stag_nfe += 1

                if self.stag_nfe >= self.maxstag:
                    break
        return self.nfe

    def get_all_data(self):
        return np.array(self.data_X), np.array(self.data_Y)

    def get_best_data(self):
        X, Y = self.get_all_data()
        return X[np.argmin(Y), :], np.min(Y)