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


minus_infinity = -99999999.0 
plus_infinity  =  99999999.0  


# Anderson problem 
# domain_X_lb = np.array([-1.0, -1.0, -1.0])
# domain_X_ub = np.array([1.0, 1.0, 1.0])
# domain_U = [[0, 2], [0, 2]]
# domain_omega = []
# for i in range(3):
#   for j in range(len(domain_U)):
#     domain_omega.append(domain_U[j])


# DACC Problem 
# domain_X_lb = np.array([0.0, minus_infinity])
# domain_X_ub = np.array([1.0, plus_infinity])
# k = 3
# domain_U = [[-1/np.pi, 1/np.sqrt(np.pi)]] * k
# m = 2
# domain_omega = domain_U * m


# AXY2 CSIP Problem (x in 2-ball)
a = np.array([0.001*np.cos(np.pi/4), 0.001*np.sin(np.pi/4)])
domain_X_lb = np.array([minus_infinity, minus_infinity])
domain_X_ub = np.array([plus_infinity, plus_infinity])
domain_U = [[0 + np.sqrt(np.pi), 2*np.pi + np.sqrt(np.pi)]] 
m = 2
domain_omega = domain_U * 2


# AXY3 CSIP Problem (x in 3-ball)
# domain_X_lb = np.array([minus_infinity, minus_infinity, minus_infinity])
# domain_X_ub = np.array([plus_infinity, plus_infinity, plus_infinity])
# domain_U = [[0, np.pi], [0, 2*np.pi]] 
# m = 3
# domain_omega = domain_U * m


print(f"The domain of u is {domain_U}")
print(f"The domain of omega is {domain_omega}")