# coding: utf-8
import numpy as np
import copyreg, copy, types, pickle
from copy import deepcopy
import cmaes


class CMAES:
    # covariance matrix adaptation - evolution strategies
    def __init__(self, prob, eval, hyperparam=None, rnd_seed=0):
        if rnd_seed:
            np.random.seed(rnd_seed)
        self.prob = prob
        self.eval = eval
        self.data_X = []
        self.data_Y = []
        self.xlb = self.prob.get_xlb()
        self.xub = self.prob.get_xub()
        self.dim = self.prob.get_xdim()
        self.xbest = None
        self.ybest = 1e25
        self.nfe = 0
        self.stag_nfe = 0
        self.ps = hyperparam['popsize']
        self.maxnfe = hyperparam['maxnfe']
        self.maxstag = hyperparam['maxstag']

    def initialize(self, m=None, s=None, pool=None):
        mu, sigma = m, s
        if mu is None:
            mu = 0.5 * (self.xlb + self.xub)
        if sigma is None:
            sigma = 1.
        if self.ps is None:
            self.hdl = cmaes.CMA(mean=mu, sigma=sigma, bounds=np.array([self.xlb, self.xub]).T, )
            self.ps = self.hdl.population_size
        else:
            self.hdl = cmaes.CMA(mean=mu, sigma=sigma, bounds=np.array([self.xlb, self.xub]), population_size=self.ps)

    def iterate(self, pool=None, verbose=0):
        gen = 0
        while self.nfe < self.maxnfe and self.stag_nfe < self.maxstag:
            gen += 1
            XY = []
            X = []
            for i in range(self.ps):
                X.append(self.hdl.ask())
            X = np.clip(np.array(X), self.xlb, self.xub)

            Y = self.eval.evaluate(self.prob, X, pool=pool)
            self.nfe += self.ps
            # print(X.shape, Y.shape, self.ps)

            for i in range(self.ps):
                XY.append((X[i,:], Y[i]))
                self.data_X.append(X[i,:])
                self.data_Y.append(Y[i])
                if Y[i] < self.ybest:
                    self.xbest = copy.deepcopy(X[i,:])
                    self.ybest = copy.deepcopy(Y[i])
                    self.stag_nfe = 0
                else:
                    self.stag_nfe += 1

            self.hdl.tell(XY)
            if verbose == 1:
                if gen % 10 == 0:
                    print(f'#gen{gen}: y*={self.ybest}, x*={self.xbest}, stag_nfe={self.stag_nfe}')

    def get_all_data(self):
        return np.array(self.data_X), np.array(self.data_Y)

    def get_best_data(self):
        X, Y = self.get_all_data()
        return X[np.argmin(Y), :], np.min(Y)


class GA:
    # Genetic Algorithm
    def __init__(self, prob, eval, maxnfe, rnd_seed=0):
        if rnd_seed:
            np.random.seed(rnd_seed)
        self.prob = prob
        self.eval = eval
        self.data_X = []
        self.data_Y = []
        self.ps = 50  # population size
        self.xlb = self.prob.xlb
        self.xub = self.prob.xub
        self.dim = self.prob.dim
        self.nfe = 0
        self.maxnfe = maxnfe

    def initialize(self, pool=None):
        self.pop = np.random.uniform(self.xlb, self.xub, (self.ps, self.dim))
        self.fit = self.eval.evaluate(self.prob, self.pop, pool=pool)
        self.nfe += self.ps
        for i in range(self.ps):
            self.data_X.append(deepcopy(self.pop[i, :]))
            self.data_Y.append(self.fit[i])
        self.gbx = deepcopy(self.pop[np.argmin(self.fit), :])
        self.gby = np.min(self.fit)

    def polynomial_mutation(self, individual, eta=20):
        mutant = np.copy(individual)
        for i in range(len(mutant)):
            if np.random.rand() < 1 / self.dim:  # mutation probability
                delta = 1 - np.random.rand()
                if delta < 0.5:
                    delta_q = (2 * delta) ** (1 / (eta + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - delta)) ** (1 / (eta + 1))
                mutant[i] += delta_q * (self.xub[i] - self.xlb[i])
                mutant[i] = np.clip(mutant[i], self.xlb[i], self.xub[i])
        return mutant

    def simulated_binary_crossover(self, p1, p2, eta=15):
        u = np.random.rand(len(p1))
        beta = np.zeros_like(u)
        beta[u <= 0.5] = (2 * u[u <= 0.5]) ** (1 / (eta + 1))
        beta[u > 0.5] = (1 / (2 * (1 - u[u > 0.5]))) ** (1 / (eta + 1))
        child1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        child2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return child1, child2

    def iterate(self, pool=None, verbose=0):
        while self.nfe < self.maxnfe:
            if verbose == 1:
                print("nfe", self.nfe, "gbfit", self.gby)

            # Selection - Tournament Selection
            new_pop = [deepcopy(self.pop[np.argmin([self.fit[np.random.randint(0, self.ps)] for _ in range(2)])]) for _
                       in range(self.ps)]

            # Crossover - Simulated Binary Crossover
            for i in range(0, self.ps, 2):
                if i + 1 < self.ps:
                    new_pop[i], new_pop[i + 1] = self.simulated_binary_crossover(new_pop[i], new_pop[i + 1])

            # Mutation - Polynomial Mutation
            new_pop = [self.polynomial_mutation(ind) for ind in new_pop]

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

    def get_all_data(self):
        return np.array(self.data_X), np.array(self.data_Y)

    def get_best_data(self):
        X, Y = self.get_all_data()
        return X[np.argmin(Y), :], np.min(Y)


class DE(GA):
    # differential evolution
    def __init__(self, prob, eval, maxnfe, rnd_seed=0):
        super().__init__(prob, eval, maxnfe, rnd_seed=rnd_seed)
        self.ps = 20

    def iterate(self, pool=None, verbose=0):
        while self.nfe < self.maxnfe:
            if verbose == 1:
                print("nfe", self.nfe, "gbfit", self.gby)

            CR = 0.9
            F = 0.5

            new_pop = deepcopy(self.pop)
            for i in range(self.ps):
                indices = np.random.choice([x for x in range(self.ps) if x != i], 3, replace=False)
                a, b, c = indices

                mutant = self.pop[a] + F * (self.pop[b] - self.pop[c])
                mutant = np.clip(mutant, self.xlb, self.xub)

                trial = deepcopy(self.pop[i])
                cross_points = np.random.rand(self.dim) < CR
                trial[cross_points] = mutant[cross_points]

                if not np.any(cross_points):
                    rnd_dim = np.random.randint(0, self.dim)
                    trial[rnd_dim] = mutant[rnd_dim]

                new_pop[i] = deepcopy(trial)

            new_fit = self.eval.evaluate(self.prob, new_pop, pool=pool)
            self.nfe += self.ps

            for i in range(self.ps):
                self.data_X.append(deepcopy(new_pop[i, :]))
                self.data_Y.append(new_fit[i])
                # selection
                if new_fit[i] < self.fit[i]:
                    self.fit[i] = new_fit[i]
                    self.pop[i, :] = deepcopy(new_pop[i, :])
                    if new_fit[i] < self.gby:
                        self.gby = new_fit[i]
                        self.gbx = deepcopy(new_pop[i, :])

        return self.nfe


class PSO:
    # particle swarm optimization
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

    def initialize(self, pool=None):
        self.pos = np.random.uniform(self.xlb, self.xub, (self.ps, self.dim))
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
        best_index = 0
        for i in range(len(slices)):
            best_index = np.argmin(Y[:slices[i] + 1])
            retX.append(X[best_index, :])
            retY.append(Y[best_index])
        retX, retY = np.array(retX), np.array(retY)
        return retX, retY


class CLPSO(PSO):
    '''
        Inplementation of "Comprehensive learning particle swarm optimizer for global optimization of multimodal functions"
        in IEEE TEVC.
    '''
    def __init__(self, prob, eval, maxnfe, maxstag=1e25, rnd_seed=0):
        super().__init__(prob, eval, maxnfe, maxstag=maxstag, rnd_seed=rnd_seed)

    def initialize(self, pool=None):
        self.pos = np.random.uniform(self.xlb, self.xub, (self.ps, self.dim))
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
        self.m = 7  # refreshing gap
        self.c = 1.49445  # acceleration coefficient

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
                    self.vel[i, d] = w * self.vel[i, d] + self.c * np.random.random() * (self.pbx[self.fi[i, d], d] - self.pos[i, d])
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


class MELPSO(PSO):
    '''
        Implementation of "Knowledge Embedding-Assisted Multi-Exemplar Learning Particle Swarm Optimization for Traffic
        Signal Timing Optimization".
    '''
    def __init__(self, prob, eval, maxnfe, maxstag=1e25, rnd_seed=0):
        super().__init__(prob, eval, maxnfe, maxstag=maxstag, rnd_seed=rnd_seed)

    def initialize(self, pool=None):
        self.pos = np.random.uniform(self.xlb, self.xub, (self.ps, self.dim))
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
        self.m = 7  # refreshing gap
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

if __name__ == "__main__":
    # from multiprocessing import Pool
    # import distribution
    # import problem
    # import time
    # pool = Pool(20)
    # distribution.prepare('net_single1', 1)
    # pg = problem.TSTOProblemGenerator('net_single1', 1, rnd_seed=1)
    # taus = [distribution.enumerate_train_tau() for i in range(20)]
    # prob_List = [pg.get_prob_instance(tau) for tau in taus]
    # prob = prob_List[0]
    # print('taus', taus)
    # ev = problem.TrafficPerformanceEvaluator('spmi', use_pool=True, workerids=[i for i in range(20)])
    # t0 = time.time()
    # pso = PSO(prob,ev,100)
    # pso.initialize(pool=pool)
    # pso.iterate(pool=pool,verbose=1)
    # print('used time', time.time() - t0)
    pass
