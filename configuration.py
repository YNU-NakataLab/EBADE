###### configuration.py #####
#                                           Last Update:  2024/1/17
#
# File for configuration
# The instance is made and named "cnf" in main.py


# import libraries and other files
import os, shutil
import numpy            as np


class Configuration:

    """ constructor """
    # initialize method
    def __init__(self):

        """ instance variable """
        # Experimental setting
        self.do_console     = False             # whether print live score
        self.max_trial      = 21                # trial times
        self.max_evals      = 10000             # maximum number of fitness evaluation

        # Problem setting
        self.prob_dim_list  = [10, 20, 30]      # dimension list
        self.prob_dim       = 0                 # dimension

        self.prob_name      = [
            # Unimodal
            "F1",                                   # Sphere Function
            # Multimodal
            "F2",                                   # Rosenbrock
            "F3",                                   # Ackley
            "F4",                                   # Rastrigin
            "F5",                                   # Griewank
            "F6",                                   # Weierstrass
            "F7"                                    # Schwefel
            ]
            
        # CaDE setting
        self.subpopulations = 25                # M : the number of subpopulations
        self.pop_size       = 4                 # N : subpopulation size
        self.analyte        = 6                 # K : the number of analyte (candidate parameter configurations)
        self.init_variants  = [0.5, 0.9, 0, 0]  # [F, CR, ID_mutation, ID_crossover]
        self.max_gen        = int(self.max_evals/(self.subpopulations*self.pop_size))
                                                # maximum generation

        # DE setting
        self.p_pbest        = 0.5               # p : probability in current-to-pbest/1
        self.mutation       = [                 # mutation variants
            # "rand/1",                               # 0
            # "rand/2",                               # 1
            "best/1",                               # 2
            # "best/2",                               # 3
            # "current-to-rand/1",                    # 4
            "current-to-best/1",                    # 5
            "current-to-pbest/1",                   # 6
            "rand-to-best/1"                        # 7
            ]
        self.crossover      = [                 # crossover variants
            "binomial",                             # 0
            "exponential"                           # 1
            ]


    """ instance method """
    # set random seed of "numpy.random"
    def setRandomSeed(self, seed=0):
        self.seed = seed
        self.rd = np.random
        self.rd.seed(self.seed)

    # I/O setting
    def makeOutDirectory(self, dim):

        alg_name = "EBADE_K{}-M{}".format(self.analyte, self.subpopulations)

        # where log generated
        self.path_out       = "./_log/{}_D{}".format(alg_name, dim)
        
        prob_list = list(range(len(self.prob_name)))
        if not os.path.isdir(self.path_out):    # if not exist
            os.makedirs(self.path_out)          # make it
        else:
            for prob in range(len(self.prob_name)):
                if os.path.isfile(self.path_out + "/" + self.prob_name[prob] + "/statistics.csv"):
                    prob_list.remove(prob) 

        if not (len(prob_list) == 0):
            # save configuraion
            shutil.copy("./configuration.py", self.path_out + "/_log_setting.txt")

        return prob_list