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



# Define the custom partition class with K=6
class CustomKaryPartition(KaryPartition):
    def __init__(self, domain, node=P_node):
        super().__init__(domain=domain, K=5, node=node)  # Set K=6

class AdaptiveKaryPartition(KaryPartition):
    def __init__(self, domain, node=P_node):
        super().__init__(domain=domain, K=2, node=node)  # Set K=6
        self.split_dim = 0
    
    def update_K(self, new_K):
        self.K = new_K

    def get_K(self): 
        return(self.K) 
    
    def get_split_dim(self):
        return self.split_dim

    def set_split_dim(self, dim):
        self.split_dim = dim 

    # Rewrite the make_children function in the Partition class
    def make_children(self, parent, newlayer=False):
        """
        The function to make children for the parent node with a standard K-ary partition, i.e., split every
        parent node into K children nodes of the same size. If there are multiple dimensions, the dimension to split the
        parent is chosen randomly

        Parameters
        ----------
        parent:
            The parent node to be expanded into children nodes

        newlayer: bool
            Boolean variable that indicates whether or not a new layer is created

        Returns
        -------

        """

        parent_domain = parent.get_domain()
        dim = np.random.randint(0, len(parent_domain))
        # dim = self.split_dim
        selected_dim = parent_domain[dim]

        new_nodes = []
        boundary_points = np.linspace(selected_dim[0], selected_dim[1], num=self.K + 1)
        for i in range(self.K):
            domain = copy.deepcopy(parent_domain)
            domain[dim] = [boundary_points[i], boundary_points[i + 1]]
            node = self.node(
                depth=parent.get_depth() + 1,
                index=self.K * parent.get_index() - (self.K - i - 1),
                parent=parent,
                domain=domain,
            )
            new_nodes.append(node)

        parent.update_children(new_nodes)

        if newlayer:
            self.node_list.append(new_nodes)
            self.depth += 1
        else:
            self.node_list[parent.get_depth() + 1] += new_nodes
 
    


