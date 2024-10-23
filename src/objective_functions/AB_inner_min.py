from mosek.fusion import * 
import mosek.fusion.pythonic
import numpy as np 
from scipy.optimize import minimize

def compute_mosek_constraint_anderson(x, u1, u2): 
    compute_u_term = (-1.0/6)*(((u1 - 1)**2 + u2)*(u1 + (2 - u2)))
    return (Expr.add(Expr.mul(-1.0*x[0], u1), \
                 Expr.add((Expr.mul(-1.0*x[1], u2)), \
                          Expr.add(-1.0*x[2], compute_u_term))) <= 0) 
    
def compute_objective_anderson(x): 
    return x[2]

def mosek_minimize_anderson(omega, domain_X_lb, domain_X_ub): 
    #Create an environment in Mosek
    with Model('model') as M:
        index1 = 0
        index2 = 1 
        x = M.variable('x', 3, Domain.inRange(domain_X_lb, domain_X_ub))
        dict_constraints = {1: '1c', 2: '2c', 3:'3c'}
        for i in dict_constraints.keys(): 
            u1 = omega[index1]
            u2 = omega[index2]
            M.constraint(dict_constraints[i], compute_mosek_constraint_anderson(x, u1, u2))
            index1 += 2
            index2 += 2
        M.objective("obj", ObjectiveSense.Minimize, compute_objective_anderson(x))
        #M.setSolverParam("intpntTolRelGap", 1)  # Relative gap tolerance
        #M.setSolverParam("intpntTolPrimalRel", 1)  # Primal feasibility tolerance
        #M.setSolverParam("intpntTolDualRel", 1)  # Dual feasibility tolerance
        M.solve()
        solx = x.level()
        return (compute_objective_anderson(solx), list(solx)) 


def compute_mosek_constraint_DACC(x, u_list): 
    lhs = x[0]*np.linalg.norm(u_list, ord = np.inf) \
    - np.linalg.norm(u_list, ord = np.inf)**2 - x[1]
    return (Expr.add(0, lhs) <= 0 ) 
    
def compute_objective_DACC(x): 
    return x[1]

def mosek_minimize_DACC(omega, k, domain_X_lb, domain_X_ub): 
    #Create an environment in Mosek
    with Model('model') as M:
        index1 = 0
        index2 = k 
        x = M.variable('x', 2, Domain.inRange(domain_X_lb, domain_X_ub))
        dict_constraints = {1: '1c', 2: '2c'}
        for i in dict_constraints.keys(): 
            u_list = omega[index1:index2]
            M.constraint(dict_constraints[i], compute_mosek_constraint_DACC(x, u_list))
            index1 += k
            index2 += k
        M.objective("obj", ObjectiveSense.Minimize, compute_objective_DACC(x))
        #M.setSolverParam("intpntTolRelGap", 1)  # Relative gap tolerance
        #M.setSolverParam("intpntTolPrimalRel", 1)  # Primal feasibility tolerance
        #M.setSolverParam("intpntTolDualRel", 1)  # Dual feasibility tolerance
        M.solve()
        solx = x.level()
        return (compute_objective_DACC(solx), list(solx)) 

def compute_mosek_constraint_AXY2(x, u_list): 
    y1 = np.cos(u_list)
    y2 = np.sin(u_list)
    return (Expr.add(Expr.mul(x[0], y1),Expr.mul(x[1], y2)) <= 1 ) 
    
def compute_objective_AXY2(x, a = np.array([0, 0])):
    a_expr = Expr.constTerm(a)
    # Define the expression for (x - a)

    x_minus_a = Expr.sub(x, a_expr)
    # Compute the Euclidean norm (2-norm) of (x - a)
    norm_x_minus_a = Expr.dot(x_minus_a, x_minus_a)
    return norm_x_minus_a

def mosek_minimize_AXY2(omega, domain_X_lb, domain_X_ub, a): 
    #Create an environment in Mosek
    with Model('model') as M:
        index = 0
        x = M.variable('x', 2, Domain.inRange(domain_X_lb, domain_X_ub))
        dict_constraints = {1: '1c', 2: '2c'}
        for i in dict_constraints.keys(): 
            u_list = omega[index]
            M.constraint(dict_constraints[i], compute_mosek_constraint_AXY2(x, u_list))
            index += i
        M.objective("obj", ObjectiveSense.Minimize, compute_objective_AXY2(x, a))
        #M.setSolverParam("intpntTolRelGap", 1)  # Relative gap tolerance
        #M.setSolverParam("intpntTolPrimalRel", 1)  # Primal feasibility tolerance
        #M.setSolverParam("intpntTolDualRel", 1)  # Dual feasibility tolerance
        M.solve()
        solx = x.level()
        return (compute_objective_AXY2(solx, a), list(solx)) 

def scipy_minimize_AXY2(omega, domain_X_lb, domain_X_ub, a):

    def objective_function(x):
      return np.linalg.norm(x - a)

    def bounds_function(i):
      return (domain_X_lb[i], domain_X_ub[i])

    def constraint_function1(x, k):
      index  =  k 
      y1 = np.cos(omega[index])
      y2 = np.sin(omega[index])
      constraint_k = x[0]*y1 + x[1]*y2 - 1 
      return constraint_k

    bounds = [bounds_function(i) for i in range(2)] #minimise

    constraint_functions1 = [lambda x, k=k: -constraint_function1(x, k) for k in range(2)]

    constraints = [{'type': 'ineq', 'fun': func} for func in constraint_functions1]

    initial_guess = [0]*2

    result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, \
                      options={'disp': False,  'ftol': 1e-5 })

    if result.success == True:
      return result.fun, result.x
    elif result.success == False:
      print(f"Failed to solve inner optimization. Error message: f{result.message}")
      return 1e8, result.x
