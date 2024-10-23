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
from AA_objective_functions import * 
from AA_parameters import * 
from AA_custom_partitions import * 

    
T  = 1000
algo = SOO(n=T, h_max=np.sqrt(T), domain=domain_omega, partition=AdaptiveKaryPartition)
#print(f"The directory of the algorithm is {dir(algo)}")
target = AXY2()

list_depths              = []
list_iteration_numbers   = []
list_sampled_point_omega = []
list_optimal_point_x     = []
list_rewards             = []
list_cumulative_reward   = []
list_average_reward      = []
list_cumulative_regret   = []
list_best_reward         = []
list_point_and_reward    = []
cum_reward = 0
average_reward = 0
start_time = time.time()
# either for-loop or while-loop

for t in range(1, T+1):
    # Append the current iteration number
    #Pull a point in omega space
    point = algo.pull(t)
    curr_node = algo.curr_node
    
    #print(dir(curr_node))
    

    # # Run the optimization for that omega
    # inf_G_omega, infimizer_x = target.f(omega=point, domain_omega= domain_omega, \
    #                                     domain_U=domain_U, \
    #                                       domain_X_lb=domain_X_lb, \
    #                                         domain_X_ub=domain_X_ub, k = k )
    
    inf_G_omega, infimizer_x = target.f(omega=point, domain_omega= domain_omega, \
                                    domain_U=domain_U, \
                                      domain_X_lb=domain_X_lb, \
                                        domain_X_ub=domain_X_ub, a = a)
    reward = inf_G_omega
    curr_node.update_reward(inf_G_omega) 
    # Update the rewards and regrets
    average_reward = average_reward*1.0*((t - 1)/t) + reward/t
    cum_reward     += reward
    # depth_k = curr_node.depth
    # algo.partition.update_K(int(depth_k + 1))

    # if(t%200 == 0): 
    #    curr_K = algo.partition.get_K()
    #    print(f"old K is {curr_K}")
    #    curr_K = min(16, int(curr_K*2))
    #    algo.partition.update_K(curr_K)
    #    print(f"new K is {curr_K}")
    # Update all the lists
    if(t%10000 == 0):
      print(f"At iteration number: {t}")
      print(f"Cumulative runtime: {time.time() - start_time}")
    
      
    if(t%1 == 0): 
      list_depths.append(curr_node.depth) 
      list_iteration_numbers.append(t)
      list_sampled_point_omega.append(point)
      list_rewards.append(inf_G_omega)
      list_point_and_reward.append((curr_node.depth, curr_node.index, \
                                    [round(x, 3) for x in point], \
                              infimizer_x, \
                                    curr_node.reward)) 
      list_optimal_point_x.append(infimizer_x)
      list_average_reward.append(average_reward)
      list_cumulative_reward.append(cum_reward)
      if (len(list_best_reward) > 0): 
        old_best = max(list_best_reward)
      list_best_reward.append(max(list_rewards))
      new_best = max(list_best_reward)
      if (len(list_best_reward) > 1): 
        if (new_best > old_best):
          print(f"Spike observed at iteration number {t}")
      #There is some BUG here!
      list_cumulative_regret.append(t*target.fmax - cum_reward)

    # Update the algorithm
    algo.receive_reward(t, reward)
    # if (t == 999):
    #   print(f"Point sampled is {point} with class {point.__class__}")
    #   print(f"Reward returned is {reward} with class {reward.__class__}")

with open('points_and_rewards.txt', 'w') as f:
    for line in list_point_and_reward:
        f.write(f"{line}\n")
plt.plot(list_iteration_numbers, list_best_reward)
#sns.plot(list_iteration_numbers, infimizer_x)
plt.show()