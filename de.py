###### de.py #####
#                                           Last Update:  2024/1/17
#
# File for the algorithm of Differential Evolution
# The instance is made and named "alg[i]" in EmulationBasedAdaptiveDifferentialEvolution class


# import libraries and other files
import numpy            as np


class DifferentialEvolution:

    """ constructor """
    # initialize method
    def __init__(self, cnf, fnc, variants):
        self.cnf        = cnf       # configuration instance
        self.fnc        = fnc       # function      
        self.pop        = []        # Pi = {x_1 x_2 ... x_(pop_size}    : population set
        self.fit        = []        # Fi = {f(x_1) f(x_2) ... f(x_ )}   : fitness set
        self.grad       = []        # Gi = {g(x_1) g(x_2) ... g(x_ )}   : Fitness Improvement Rate (FIR) set

        self.selector   = DEVariant(self.cnf, self.fnc)   # set mutation and crossover
        self.setVariants(variants)



    """ instance method """
    # set variants to the subpopulation
    def setVariants(self, variants):
        self.F              = variants[0]
        self.CR             = variants[1]
        self.v_MUTATION     = variants[2]
        self.mutation       = self.selector.mutation_map[self.v_MUTATION]
        self.v_CROSSOVER    = variants[3] 
        self.crossover      = self.selector.crossover_map[self.v_CROSSOVER]

    # initialize solutions
    def initializePopulation(self):
        for i in range(self.cnf.pop_size):
            # generate randomly
            if hasattr(self.fnc, "init_range"):
                self.pop.append(self.cnf.rd.uniform(self.fnc.init_range[0], self.fnc.init_range[1]))
            else:
                self.pop.append(self.cnf.rd.uniform(self.fnc.axis_range[0], self.fnc.axis_range[1]))
            self.fit.append(self.fnc.doEvaluate(self.pop[i]))
            self.grad.append(0)

    # run DE
    def run(self, ead_pop, ead_fit, b1):
        self.pop, self.fit, self.grad  = self._generationShift(self.pop, self.fit, ead_pop, ead_fit, b1, self.grad)

    # generational shift
    def _generationShift(self, pop, fit, ead_pop, ead_fit, _b1, grd):
        _pb1    = None  # x_p-best  = [x1 x2 ... xD]   : one of better solutions (top ~ subpopulations*pop_size*p_pbest)
        if self.v_MUTATION in [2]: # CHANGE
            _tmp, _max = np.argsort(ead_fit), (int)(max(2.0, self.cnf.subpopulations * self.cnf.pop_size * self.cnf.p_pbest))
            _pb1 = ead_pop[self.cnf.rd.choice(_tmp[0:_max])]

        _v      = []    # mutant solution
        _u      = []    # crossovered solution
        _e      = []    # the fitness of _u
        for i in range(self.cnf.pop_size):
            _v.append(self.mutation(self.F, ead_pop, _b1, pop[i], _pb1))    # mutation
            _u.append(self.crossover(self.CR, pop[i], _v[i]))               # crossover
            _e.append(self.fnc.doEvaluate(_u[i]))                           # evaluate _u

        # selection
        for i in range(self.cnf.pop_size):
            grd[i] = (_e[i]-fit[i])/fit[i]
            if _e[i] <= fit[i]:
                pop[i], fit[i] = np.copy(_u[i]), _e[i]
        return pop, fit, grd

    # at priorValidation component, make a set of next-generation solutions to test a candidate configuration
    def testVariants(self, variants, pop, fit, ead_pop, ead_fit, _b1, target):
        self.setVariants(variants)
        _pb1    = None  # x_p-best  = [x1 x2 ... xD]   : one of better solutions (top ~ subpopulations*pop_size*p_pbest)
        if self.v_MUTATION in [2]: # CHANGE
            _tmp, _max = np.argsort(ead_fit), (int)(max(2.0, self.cnf.subpopulations * self.cnf.pop_size * self.cnf.p_pbest))
            _pb1 = ead_pop[self.cnf.rd.choice(_tmp[0:_max])]

        _v      = []    # mutant solution
        _ret    = []    # temporary _u
        for i in range(self.cnf.pop_size):
            _v.append(self.mutation(self.F, ead_pop, _b1, pop[i], _pb1))    # mutation
            _ret.append(self.crossover(self.CR, pop[i], _v[i]))             # crossover
        
        return _ret

    # get the best solution and its fitness
    def __getBestSolution(self, pop, fit):
        _argmin = np.argmin(fit)
        return pop[_argmin], fit[_argmin]



class DEVariant:

    """ constructor """
    # initialize method
    def __init__(self, cnf, fnc):

        """ instance variable """
        self.cnf = cnf
        self.fnc = fnc

        self.mutation_map = [
            # self.mutation_rand1,            # 0 : rand/1  
            # self.mutation_rand2,            # 1 : rand/2  
            self.mutation_best1,            # 2 : best/1  
            # self.mutation_best2,            # 3 : best/2  
            # self.mutation_crand1,           # 4 : current-to-rand/1
            self.mutation_cbest1,           # 5 : current-to-best/1
            self.mutation_cpbest1,          # 6 : current-to-pbest/1
            self.mutation_rbest1            # 7 : rand-to-best/1
        ]

        self.crossover_map = [
            self.crossover_binomial,        # 0 : binomial
            self.crossover_exponential,     # 1 : exponential
        ]


    """ instance method """
    # *** mutation ***
    # 0 : rand/1  
    def mutation_rand1(self, F, pop, b1, n1, pb1):
        tmp = list(range(len(pop)))     # get an index list
        tmp.remove(np.argmax(np.bincount(np.where(pop==n1)[0])))
        _r1, _r2, _r3 = self.cnf.rd.choice(tmp, 3)
        r1, r2, r3 = pop[_r1], pop[_r2], pop[_r3]
        return r1 + F * (r2 - r3)

    # 1 : rand/2
    def mutation_rand2(self, F, pop, b1, n1, pb1):
        tmp = list(range(len(pop)))
        tmp.remove(np.argmax(np.bincount(np.where(pop==n1)[0])))
        _r1, _r2, _r3, _r4, _r5 = self.cnf.rd.choice(tmp, 5)
        r1, r2, r3, r4, r5 = pop[_r1], pop[_r2], pop[_r3], pop[_r4], pop[_r5]
        return r1 + F * (r2 - r3) + F * (r4 - r5)

    # 2 : best/1
    def mutation_best1(self, F, pop, b1, n1, pb1):
        tmp = list(range(len(pop)))
        tmp.remove(np.argmax(np.bincount(np.where(pop==n1)[0])))
        if not (n1 == b1).all(): tmp.remove(np.argmax(np.bincount(np.where(pop==b1)[0])))
        _r1, _r2 = self.cnf.rd.choice(tmp, 2)
        r1, r2 = pop[_r1], pop[_r2]
        return b1 + F * (r1 - r2)

    # 3 : best/2
    def mutation_best2(self, F, pop, b1, n1, pb1):
        tmp = list(range(len(pop)))
        tmp.remove(np.argmax(np.bincount(np.where(pop==n1)[0])))
        if not (n1 == b1).all(): tmp.remove(np.argmax(np.bincount(np.where(pop==b1)[0])))
        _r1, _r2, _r3, _r4 = self.cnf.rd.choice(tmp, 4)
        r1, r2, r3, r4 = pop[_r1], pop[_r2], pop[_r3], pop[_r4]
        return b1 + F * (r1 - r2) + F * (r3 - r4)

    # 4 : current-to-rand/1
    def mutation_crand1(self, F, pop, b1, n1, pb1):
        tmp = list(range(len(pop)))
        tmp.remove(np.argmax(np.bincount(np.where(pop==n1)[0])))
        _r1, _r2, _r3 = self.cnf.rd.choice(tmp, 3)
        r1, r2, r3 = pop[_r1], pop[_r2], pop[_r3]
        return n1 + F * (r1 - n1) + F * (r2 - r3)

    # 5 : current-to-best/1
    def mutation_cbest1(self, F, pop, b1, n1, pb1):
        tmp = list(range(len(pop)))
        tmp.remove(np.argmax(np.bincount(np.where(pop==n1)[0])))
        if not (n1 == b1).all(): tmp.remove(np.argmax(np.bincount(np.where(pop==b1)[0])))
        _r1, _r2 = self.cnf.rd.choice(tmp, 2)
        r1, r2 = pop[_r1], pop[_r2]
        return n1 + F * (b1 - n1) + F * (r1 - r2)

    # 6 : current-to-pbest/1
    def mutation_cpbest1(self, F, pop, b1, n1, pb1):
        tmp = list(range(len(pop)))
        tmp.remove(np.argmax(np.bincount(np.where(pop==n1)[0])))
        if not (n1 == b1).all(): tmp.remove(np.argmax(np.bincount(np.where(pop==b1)[0])))
        _r1, _r2 = self.cnf.rd.choice(tmp, 2)
        r1, r2 = pop[_r1], pop[_r2]
        return n1 + F * (pb1 - n1) + F * (r1 - r2)

    # 7 : rand-to-best/1
    def mutation_rbest1(self, F, pop, b1, n1, pb1):
        tmp = list(range(len(pop)))
        tmp.remove(np.argmax(np.bincount(np.where(pop==n1)[0])))
        if not (n1 == b1).all(): tmp.remove(np.argmax(np.bincount(np.where(pop==b1)[0])))
        _r1, _r2, _r3 = self.cnf.rd.choice(tmp, 3)
        r1, r2, r3 = pop[_r1], pop[_r2], pop[_r3]
        return r1 + F * (b1 - r1) + F * (r2 - r3)

    # *** crossover ***
    # 0 : binomial
    def crossover_binomial(self, CR, x, v):
        rmat = self.cnf.rd.rand(self.cnf.prob_dim) < CR
        rmat[self.cnf.rd.randint(self.cnf.prob_dim)] = True
        u = np.copy(x)
        u[rmat] = v[rmat]
        u = np.clip(u, self.fnc.axis_range[0], self.fnc.axis_range[1])
        return u

    # 1 : exponential
    def crossover_exponential(self, CR, x, v):
        k = 1
        u = np.copy(x)
        j = self.cnf.rd.randint(self.cnf.prob_dim)
        while True:
            u[j] = v[j]
            j = (1 + j) % self.cnf.prob_dim
            k += 1
            if not((self.cnf.rd.rand() < CR) and (k < self.cnf.prob_dim)):
                u = np.clip(u, self.fnc.axis_range[0], self.fnc.axis_range[1])
                return u


