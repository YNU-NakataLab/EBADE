###### function.py #####
#                                           Last Update:  2024/1/17
#
# File for Benchmark Suite (Test Problems)
# The instance is made and named "fnc" in main.py


# import libraries and other files
import numpy    as np
from distutils.version import StrictVersion


class Function:

    """ constructor """
    # initialize method
    def __init__(self, prob_dim:int=None, prob_name=None):

        """ instance variable """
        self.prob_name   = prob_name    # probrem name
        self.prob_dim    = prob_dim     # dimension
        self.axis_range  = []           # domain of each axis
        self.evaluate    = None         # alternative function
        self.total_evals = 0            # FEs: the number of fitness evaluations
        self.total_gen   = 0

    """ instance method """
    # evaluate
    def doEvaluate(self, x):
        self.total_evals += 1
        x = np.array(x)
        return self.evaluate(x)

    # reset FEs
    def resetTotalEvals(self):
        self.total_evals = 0
        self.total_gen   = 0


class Basic(Function):

    """ constructor """
    # initialize method
    def __init__(self, prob_dim:int, prob_name:str):

        """ instance variable """
        self.prob_name      = prob_name
        self.prob_dim       = prob_dim
        self.evaluate       = None
        self.total_evals    = 0
        self.total_gen      = 0
        self.axis_range     = [np.full(self.prob_dim, -100.), np.full(self.prob_dim, 100.)]

        # choice of functions
        if self.prob_name == "F1":
            self.evaluate = self.F1
        elif self.prob_name == "F2":
            self.evaluate = self.F2
            self.axis_range  = [np.full(self.prob_dim, -2.048), np.full(self.prob_dim, 2.048)]
        elif self.prob_name == "F3":
            self.evaluate = self.F3
            self.axis_range  = [np.full(self.prob_dim, -32.768), np.full(self.prob_dim, 32.768)]
        elif self.prob_name == "F4":
            self.evaluate = self.F4
            self.axis_range  = [np.full(self.prob_dim, -5.12), np.full(self.prob_dim, 5.12)]
        elif self.prob_name == "F5":
            self.evaluate = self.F5
            self.axis_range  = [np.full(self.prob_dim, -600.), np.full(self.prob_dim, 600.)]
        elif self.prob_name == "F6":
            self.evaluate = self.F6
            self.axis_range  = [np.full(self.prob_dim, -0.5), np.full(self.prob_dim, 0.5)]
        elif self.prob_name == "F7":
            self.evaluate = self.F7
            self.axis_range  = [np.full(self.prob_dim, -500.), np.full(self.prob_dim, 500.)]
        else:
            print("Error: Do not exist Function {} (function.py)".format(prob_name))
            return None

        print("\t[ Problem {} ]".format(prob_name))

        # numpy version check
        if StrictVersion(np.version.version) < StrictVersion("1.19"):
            print("Error: Numpy version >= 1.19 required. But {} now.".format(np.version.version))
            return None

    def doEvaluate(self, x):
        x = np.asarray(x)
        if len(x.shape) == 2:
            self.total_evals += x.shape[0]
            ret = []
            for i in range(x.shape[0]):
                ret.append(self.evaluate(x[i]))
            ret = np.asarray(ret)
        elif len(x.shape) == 1:
            self.total_evals += 1
            ret = self.evaluate(x)
        return ret

    def valid_arg(self, x) -> bool:
        if x.shape[-1] != self.prob_dim:
            print("Error: The argument of function {} is not a {}-d vector (function.py)"\
            .format(self.prob_name, self.prob_dim))
            return False
        elif np.any(x < self.axis_range[0]) or np.any(x > self.axis_range[1]):
            print("Error: The argument is out of bounds in function {}".format(self.prob_name))
            return False
        else:
            return True

    # Sphere
    def F1(self, x):
        ret = 0.
        for i in range(self.prob_dim):
            ret += x[i] * x[i]
        return ret

    # Rosenbrock
    def F2(self, x):
        ret = 0.
        for i in range(self.prob_dim - 1):
            ret += 100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2
        return ret

    # Ackley
    def F3(self, x):
        sum_1 = 0.
        sum_2 = 0.
        for i in range(self.prob_dim):
            sum_1 += x[i] * x[i]
            sum_2 += np.cos(2 * np.pi * x[i])
        ret = -20 * np.exp(-0.2 * np.sqrt(sum_1 / self.prob_dim)) - np.exp(sum_2 / self.prob_dim) + 20 + np.e
        return ret

    # Rastrigin
    def F4(self, x):
        ret = 0.
        for i in range(self.prob_dim):
            ret += x[i]**2 - 10 *np.cos( 2 * np.pi * x[i] ) + 10
        return ret

    # Griewank
    def F5(self, x):
        sum_1 = 0.
        prod_1 = 1.
        for i in range(self.prob_dim):
            sum_1 += x[i] * x[i]
            prod_1 *= np.cos(x[i] / np.sqrt(i + 1))
        ret = 1. - prod_1 + sum_1 / 4000.
        return ret

    # Weierstrass
    def F6(self, x):
        a, b, kmax = 0.5, 3, 20
        ret, tmp = 0., 0.
        for i in range(self.prob_dim):
            for k in range(kmax):
                ret += a**k * np.cos( 2 * np.pi * b**k * ( x[i] + 0.5 ) )
        for k in range(kmax):
            tmp += a**k * np.cos( np.pi * b**k)
        ret -= self.prob_dim * tmp
        return ret

    # Schwefel
    def F7(self, x):
        sum_1 = 0.
        for i in range(self.prob_dim):
            sum_1 += x[i] * np.sin(np.sqrt(abs(x[i])))
        ret = 418.9829 * self.prob_dim - sum_1
        return ret
