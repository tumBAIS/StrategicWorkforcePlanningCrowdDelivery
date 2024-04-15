"""
File: backward_DP.py
Author: Julius Luy
Date: November 10th 2023
Description: This module implements the exact (i.e., under full information) stochastic backward dynamic programming
approach.

Parameters
----------
problem: object
    Class that implements the MDP's environment.
method : object
    Class that contains all hyperparameter settings.
results_path : Path object
    Stores the path to which results are saved
"""

import time
import numpy as np

from BDP.psi_calculation import calculate_psi_for
from common.brute_force import brute_force
from common.save_instance import save_instance

# Stochastic backward DP
def backward_DP(problem, method,results_path):

    # Initialize state values
    vbar = np.zeros((problem.tmax + 1, problem.maxFD + 1, problem.maxCD1 + 1, problem.maxCD2 + 1))
    act = np.zeros((problem.tmax + 1, problem.maxFD + 1, problem.maxCD1 + 1, problem.maxCD2 + 1))

    # Go through all fleet sizes in final time step
    for i in range(0,problem.maxFD+1):
        for j in range(0, problem.maxCD1 + 1):
            for k in range(0, problem.maxCD2 + 1):
                i1 = np.minimum(i + problem.action_space, np.ones(problem.action_space.shape[0]) * problem.maxFD).astype(int)
                act[problem.tmax,i,j,k] = np.argmax(problem.rew[i1, j, k, problem.tmax])

    # Backward loop calculating optimal state values and actions based on t+1
    t = max(problem.tmax - 1, 0)
    while t >= 0:
        t1 = time.time()
        vec_buffer = {}
        s = np.zeros(3)

        action_space = problem.action_space

        # Loop through all possible fleet compositions and calculate optimal policy
        for i in range(0, problem.maxFD + 1):
            for j in range(0, problem.maxCD1 + 1):
                for k in range(0, problem.maxCD2 + 1):
                    s[:] = np.array([i, j, k])
                    maxAction = min(problem.maxFD-i,action_space[action_space.shape[0]-1])
                    max_act, vbar[t, i, j, k] = brute_force(i, j, k, t, problem, method, vbar, maxAction, action_space,vec_buffer)
                    act[t, i, j, k] = max_act
        t2 = time.time()-t1
        print('timestep', t, 'solved in', t2, 's')
        t = t - 1
    save_instance(vbar, act, results_path)