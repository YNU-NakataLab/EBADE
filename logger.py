###### logger.py #####
#                                           Last Update:  2024/1/17
#
# File to get log and statistics
# The instance is made and named "log" in main.py


# import libraries and other files
import os.path
import numpy            as np
import pandas           as pd


class Logger:

    """ constructor """
    # initialize method
    def __init__(self, cnf, fnc, prob_name):

        """ instance variable """
        self.cnf, self.fnc  = cnf, fnc
        self.dat            = []                        # where data temporary stored
        self.prob_name      = prob_name

        # settings of path
        self.path_out       = self.cnf.path_out
        self.path_out       += "/" + prob_name + "/"
        self.path_trial     = self.path_out + "trials"
        if not os.path.isdir(self.path_trial):          # if not exist
            os.makedirs(self.path_trial)                # make it


    """ instance method """
    # get a log of the best solution
    def logging(self, evals, gen, _time, x_best, f_best):
        if self.cnf.do_console:
            print(" live score =>\tgen: {:04}\tevals: {:06}\tfx: {}".format(gen, evals, f_best))
        _sls = [evals, gen, _time, f_best]              # the best solution(evals, gen, f)
        # _sls.extend(x_best)                             # the best solution(x)
        self.dat.append(_sls)                           # join
    
    # output csv file
    def outLog(self, evals, gen, f_best):
        _head = "evals,gen,time,fx"
        np.savetxt(self.path_trial +'/trial{}.csv'.format(self.cnf.seed), np.array(self.dat), delimiter=',', header = _head, comments = '')
        print("*** trial: {:03}  finished *** \n\tresult =>\tgen: {:04}\tevals: {:06}\tfx: {}".format(self.cnf.seed, gen, evals, f_best))
        self.dat = []                                   # refresh



class Statistics:

    """ constructor """
    # initialize method
    def __init__(self, cnf, fnc, path_out, path_dat):

        """ instance variable """
        self.path_out = path_out
        self.path_dat = path_dat
        self.cnf      = cnf
        self.fnc      = fnc


    """ instance method """
    # make a statistics file
    def outStatistics(self):
        # read all trial csv files
        df = None                                       # data frame
        for i in range(self.cnf.max_trial):
            dat = pd.read_csv(self.path_dat+'/trial{}.csv'.format(i+1), index_col = 0)
            if i == 0:
                df = pd.DataFrame({'trial{}'.format(i+1) : np.array(dat['fx'])}, index = dat.index)
            else:
                df['trial{}'.format(i+1)] = np.array(dat['fx'])
        # output csv file
        df.to_csv(self.path_out + "all_trials.csv")

        # handling (minimum, maximum, 25 percentile, median, 75 percentile, average, standard deviation）
        _min, _max, _q25, _med, _q75, _ave, _std = [], [], [], [], [], [], []
        for i in range(len(df.index)):
            dat = np.array(df.loc[df.index[i]])
            res = np.percentile(dat, [25, 50, 75])
            _min.append(dat.min())
            _max.append(dat.max())
            _q25.append(res[0])
            _med.append(res[1])
            _q75.append(res[2])
            _ave.append(dat.mean())
            _std.append(dat.std())

        # make a data frame
        _out = pd.DataFrame({
            'min' : np.array(_min),
            'q25' : np.array(_q25),
            'med' : np.array(_med),
            'q75' : np.array(_q75),
            'max' : np.array(_max),
            'ave' : np.array(_ave),
            'std' : np.array(_std)
            },index = df.index)

        # output csv file
        _out.to_csv(self.path_out + "statistics.csv")

