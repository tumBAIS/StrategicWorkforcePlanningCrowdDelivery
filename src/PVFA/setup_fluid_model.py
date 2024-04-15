"""
File: setup_fluid_model.py
Author: Julius Luy
Date: November 10th 2023
Description: This model prepares all the settings required for the fluid approximation. It then calls the
fluid approximation by calling a C++ function which is embedded using pybind11.

Parameters
----------
problem: object
    Class that implements the MDP's environment.

Returns
----------
fluid_model : object
    Implements the fluid approximation.
"""

import time
import numpy as np
import os,sys
import pandas as pd

from common.state_transfer_fct import state_transfer_fct

from pathlib import Path
current_working_dir = Path(os.getcwd())
main_project_dir = str(current_working_dir.parent.parent) # .parent
sys.path.append(main_project_dir+'/cpp/cmake-build-release/lib')


def setup_fluid_model(problem):

    import costs_cpp as reward_function

    # Initialize operational model
    c = reward_function.NPYArray3D(problem.c.shape,np.reshape(problem.c,np.prod(problem.c.shape)))
    c_empty_routing = reward_function.NPYArray2D(problem.c_empty_routing.shape, np.reshape(problem.c_empty_routing, np.prod(problem.c_empty_routing.shape)))
    route_pat = reward_function.NPYArray2D(problem.route_pat.shape,np.reshape(problem.route_pat,np.prod(problem.route_pat.shape)))
    req_pat = reward_function.NPYArray2D(problem.req_pat.shape,np.reshape(problem.req_pat, np.prod(problem.req_pat.shape)))
    mu = reward_function.NPYArray2D(problem.mu.shape, np.reshape(problem.mu, np.prod(problem.mu.shape)))
    mob_pat = reward_function.NPYArray1D(problem.mob_pat)

    gw_pat = problem.demand_arrival_patterns
    gw_pat = reward_function.NPYArray1D(gw_pat)

    c_fix = problem.c_fix
    c_Rpen = -problem.c_Rpen
    env = reward_function.GRBEnv()
    fluid_model = []

    for t in range(0,problem.tmax+1):
        demand_t = reward_function.NPYArray1D(problem.demand[t])
        fluid_model.append(reward_function.OperationalModel(env,c,c_empty_routing,c_Rpen,
                    c_fix,route_pat,req_pat,mu,demand_t,mob_pat,problem.gw_capacity,problem.od_capacity,gw_pat))
    return fluid_model