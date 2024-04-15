"""
File: brute_force.py
Author: Julius Luy
Date: November 10th 2023
Description: This function searches for the action maximizing the state value in a brute force manner.

Parameters
----------
(i,j,k,t): int
    No. of FDs, GWs, ODs and time step
vbar: float
    State value function
maxAction: int
    Maximum action per time step
actions: vector of ints
    Vector of all actions.
vec_buffer: float dict
    Caches state values for different driver states
problem: object
    Class that implements the MDP's environment.
method : object
    Class that contains all hyperparameter settings.

Returns
-------
max_act : int
    Action that maximizes state value function
max_act_val : float
    Maximized state value
"""

import time

import numpy as np
import itertools
import math
import matplotlib.pyplot as plt

from BDP.psi_calculation import calculate_psi_for

def brute_force(i,j,k,t,problem,method,vbar,maxAction,actions,vec_buffer):
    max_act_val = None
    # Go through all actions
    for q in range(0, maxAction+1):
        s0 = i + q

        # Calculate expected next value
        psi = calculate_psi_for(s0, j, k, t, problem.rew[s0, j, k, t], problem, method.gamma, vbar, vec_buffer)
        sigma = psi

        # Store action and value if bigger than in previous iteration
        if max_act_val is None:
            max_act_val = sigma
            max_act = actions[q]
        elif max_act_val < sigma:
            max_act_val = sigma
            max_act = actions[q]
    return max_act,max_act_val