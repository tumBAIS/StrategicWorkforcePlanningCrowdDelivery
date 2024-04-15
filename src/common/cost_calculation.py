"""
File: cost_calculation.py
Author: Julius Luy
Date: November 10th 2023
Description: Calculates operational costs.

Parameters
----------
problem: object
    Class that implements the MDP's environment.

Returns
---------
c,c_pen_req,c_empty_routing : Matrix of floats (of size location x location)
    Variable cost matrix for FDs, GWs, ODs (c), operational penalty costs (c_pen_req), empty FD routing costs
    (c_empty_routing)
"""

import numpy as np
def cost_calculation(problem):
    c = np.zeros((problem.loc,problem.loc,4))

    c[:,:,0] = problem.r*problem.c_d # Variable costs of FDs
    c[:, :, 1] = problem.r * problem.c_g # Variable costs of GWs
    c[:, :, 2] = problem.c_o # Variable costs of OD

    c_pen_req = problem.c_Rpen

    c_empty_routing = np.zeros((problem.loc,problem.loc))
    c_empty_routing[:,:] = c[:,:,0]
    np.fill_diagonal(c_empty_routing, np.zeros(c.shape[0]))
    return -c,-c_pen_req,-c_empty_routing