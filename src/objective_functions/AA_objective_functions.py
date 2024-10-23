from PyXAB.partition.BinaryPartition import *
from PyXAB.partition.DimensionBinaryPartition import *
from PyXAB.partition.KaryPartition import *
from PyXAB.partition.RandomBinaryPartition import *
from PyXAB.synthetic_obj.Objective import Objective
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from sympy import Array, symbols, lambdify
import time
from scipy.optimize import minimize
from AB_inner_min import * 
from PyXAB.algos.SOO import SOO


class Anderson1_unconstrained(Objective):
    def __init__(self):
        self.fmax = -1.0/3

    def f(self, omega, domain_omega, domain_U, domain_X_lb, domain_X_ub):
        inf_G_omega, infimizer_x = mosek_minimize_anderson(omega, domain_X_lb, domain_X_ub)
        return inf_G_omega, infimizer_x

class DACC_problem(Objective):
    def __init__(self):
        self.fmax = 0

    def f(self, omega, domain_omega, domain_U, domain_X_lb, domain_X_ub, k):
        inf_G_omega, infimizer_x = mosek_minimize_DACC(omega, k, domain_X_lb, domain_X_ub)
        return inf_G_omega, infimizer_x


class AXY2(Objective):
    def __init__(self):
        self.fmax = np.inf

    def f(self, omega, domain_omega, domain_U, domain_X_lb, domain_X_ub, a):
        inf_G_omega, infimizer_x = scipy_minimize_AXY2(omega, domain_X_lb, domain_X_ub, a)
        return inf_G_omega, infimizer_x