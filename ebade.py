###### ebade.py #####
#                                           Last Update:  2024/1/17
#
# File for the algorithm of Emulation-based Adaptive Differential Evolution (EBADE)
# The instance is made and named "ead" in main.py


# import libraries and other files
import time
import numpy            as np
from   scipy.spatial    import distance
from   de               import DifferentialEvolution


class EmulationBasedAdaptiveDifferentialEvolution:

    """ constructor """
    # initialize method
    def __init__(self, cnf, fnc, log):

        """ instance variable """
        self.cnf    = cnf       # configuration instance
        self.fnc    = fnc       # function instance
        self.log    = log       # logger instance

        self.pop    = []        # P = {x_1 x_2 ... x_(pop_size * n)}    : whole population set
        self.fit    = []        # F = {f(x_1) f(x_2) ... f(x_ )}        : whole fitness set
        self.ids    = []        # [0 0 ... 0 1 1 ... 1 2 2 ... ... n]   : subpopulation IDs
        self.grad   = []        # G = {g(x_1) g(x_2) ... g(x_ )}        : whole Fitness Improvement Rate (FIR) set
        self.alg    = []        # A = {DE_1 DE_2 ... DE_n}              : (ex) DEs
                                # |P| = |F| = |ids| = pop_size * n
                                # |A|               = n

        self.b1     = []        # x_best = [x1 x2 ... xD]               : the best solution
        self.bf     = np.inf    # f_best = f(x_best)                    : fitness of the best solution
        self.loser  = []        # a set of IDs of the loser configurations
        self.target = []        # chosen targets to tune configurations


    """ instance method """
    # run CaDE
    def run(self, trial):
        elapsed_time = self._initialize()
        _end = False
        while  self.fnc.total_gen < self.cnf.max_gen:
            _end, elapsed_time = self._search(elapsed_time)
            time_log = time.time()
            if not _end:
                self._posthocValidation()
                self._priorValidation()
                time_log = time.time()
            elapsed_time = time.time() - time_log + elapsed_time
        self.log.outLog(self.fnc.total_evals, self.fnc.total_gen, self.bf)

    # initialize all subpopulations, configurations and whole population
    def _initialize(self):
        self.start = time.time()
        for i in range(self.cnf.subpopulations):
            self.ids.extend([i for ij in range(self.cnf.pop_size)])
            self.cnf.init_variants[2] = self.cnf.rd.randint(len(self.cnf.mutation))
            self.cnf.init_variants[3] = self.cnf.rd.randint(len(self.cnf.crossover))
            self.alg.append(DifferentialEvolution(self.cnf, self.fnc, self.cnf.init_variants))
            self.alg[i].initializePopulation()
            self.pop.extend(self.alg[i].pop)
            self.fit.extend(self.alg[i].fit)
            self.grad.extend(self.alg[i].grad)
        self.fnc.total_gen += 1
        self.b1, self.bf = self._getBestSolution()
        elapsed_time = time.time()
        self.log.logging(self.fnc.total_evals, self.fnc.total_gen, elapsed_time-self.start, self.b1, self.bf)
        return elapsed_time

    # search : with real fitness evaluation
    def _search(self, elapsed_time):
        if self.fnc.total_gen < self.cnf.max_gen:
            time_log = time.time()
            prev_elapsed = elapsed_time
            ead_pop, ead_fit = np.copy(self.pop), np.copy(self.fit)
            self.pop, self.fit, self.grad = [], [], []
            for i in range(self.cnf.subpopulations):
                self.alg[i].run(ead_pop, ead_fit, self.b1)
                self.pop.extend(self.alg[i].pop)
                self.fit.extend(self.alg[i].fit)
                self.grad.extend(self.alg[i].grad)
            self.fnc.total_gen += 1
            self.b1, self.bf = self._getBestSolution()
            elapsed_time = time.time()
            self.log.logging(self.fnc.total_evals, self.fnc.total_gen, elapsed_time-(time_log-prev_elapsed)-self.start, self.b1, self.bf)
        else:
            return True, elapsed_time     # end
        return False, elapsed_time

    # post-hoc validation : decide winner and loser configurations
    def _posthocValidation(self):
        # get a set of original index when sort FIR in ascending order : good -> bad
        _sort   = np.argsort(self.grad)
        # decide a set of top "n" solutions (duplication of their home subpopulations : OK)
        _top_n  = [self.pop[_sort[i]] for i in range(self.cnf.subpopulations)]
        # get a set of IDs of subpopulations which can make top "n" (duplication of IDs : NG)
        _winner = np.unique([self.ids[_sort[i]] for i in range(self.cnf.subpopulations)])
        # get a set of IDs loser (complement of "_winner")
        self.loser = np.setdiff1d(self.ids, _winner)
        # choose same target for all losers
        self.target = np.array([_top_n[0]] * len(self.loser))

    # prior validation : change loser configurations so that they can make next-generation solution nearby the target 
    def _priorValidation(self):
        for i in range(len(self.loser)):
           _variant = self.__tuning_configuration(self.alg[self.loser[i]], self.target[i])  # choose
           self.alg[self.loser[i]].setVariants(_variant)                                # set

    # detail of tuning
    def __tuning_configuration(self, alg, target):
        _var = []   # [[F_1 CR_1 mut_1 xov_1] ... [_ _ _ _analyte]] : a set of temporary variants
        _nrm = []   # [score_1 score_2 ... score_analyte]           : a set of scores
        for i in range(self.cnf.analyte):
            _var.append(self.___variantSelector())
            _nrm.append(self.___score(alg.testVariants(_var[i], alg.pop, alg.fit, self.pop, self.fit, self.b1, target), target))
        _tmp = np.argmin(_nrm)
        return _var[_tmp]

    # choose variants randomly
    def ___variantSelector(self):
        _ret = []
        _ret.append(self.cnf.rd.rand())
        _ret.append(self.cnf.rd.rand())
        _ret.append(self.cnf.rd.randint(len(self.cnf.mutation)))
        _ret.append(self.cnf.rd.randint(len(self.cnf.crossover)))
        return _ret

    # get score : calculate average norm between the target and pseudo next-generation solutions
    def ___score(self, pop, target):
        _ret = np.min(np.linalg.norm(np.array(pop)-np.array(target), axis=1))
        return _ret

    # get the best solution and its fitness
    def _getBestSolution(self):
        _argmin = np.argmin(self.fit)
        return self.pop[_argmin], self.fit[_argmin]




