###### main.py #####
#                                           Last Update:  2024/1/17
#
# File for running program
# "Kei Nishihara and Masaya Nakata, Emulation-based Adaptive Differential Evolution: Fast and Auto-tunable Approach for Moderately Expensive Optimization Problems, Complex & Intelligent Systems, accepted."


# import libraries and other files
import ebade            as eb
import configuration    as cf
import function         as fc 
import logger           as lg


""" main """
# run firstly
if __name__ == '__main__':
    cnf = cf.Configuration()
    for dim in cnf.prob_dim_list: 
        cnf.prob_dim = dim
        prob_list = cnf.makeOutDirectory(dim)
        for prob in prob_list:
            fnc = fc.Basic(cnf.prob_dim, cnf.prob_name[prob])
            log = lg.Logger(cnf, fnc, cnf.prob_name[prob])
            for trial in range(cnf.max_trial):
                fnc.resetTotalEvals()
                cnf.setRandomSeed(trial + 1)
                ead = eb.EmulationBasedAdaptiveDifferentialEvolution(cnf, fnc, log)
                ead.run(trial)

            # make a statistics file of each problem
            sts = lg.Statistics(cnf, fnc, log.path_out, log.path_trial)
            sts.outStatistics()                                        