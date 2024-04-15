"""
File: state_transfer_fct.py
Author: Julius Luy
Date: November 10th 2023
Description: This function implements the transfer function from state at time step t to the state at time step t+1.

Parameters
----------
state_t: Vector of ints
    State in time step t.
problem: object
    Class that implements the MDP's environment.
t : float
    Current time step
fluid_model : object
    Fluid model of the operational level
sCD_buffer : dict
    Stores the share of not matched CDs in for every state in time step t.

Returns
----------
state_t_prime : Vector of int
    State in time step t+1
"""

import numpy as np
import scipy.special

def state_transfer_fct(state_t,problem,t,fluid_model,sCD_buffer):

    state_t_prime = np.zeros(3)
    i1 = state_t[0].astype(int)
    i2 = state_t[1].astype(int)
    i3 = state_t[2].astype(int)


    if problem.resig_gw!=1:
            p_tmp_CD1_resig = problem.resig_gw
            p_tmp_CD2_resig = problem.resig_od
    else:
        if (i1,i2,i3) in sCD_buffer:
            p_tmp_CD1_resig = problem.p_resig_cd1_high*sCD_buffer[i1,i2,i3][0]+(1-sCD_buffer[i1,i2,i3][0])*problem.p_resig_cd1_low
            p_tmp_CD2_resig =  problem.p_resig_cd2_high*sCD_buffer[i1,i2,i3][1]+(1-sCD_buffer[i1,i2,i3][1])*problem.p_resig_cd2_low
        else:
            output = fluid_model[t].solve_model(i1,i2,i3)
            sCD1 = output.s_g
            sCD2 = output.s_o
            sCD_buffer[i1, i2, i3] = [sCD1,sCD2]
            p_tmp_CD1_resig = problem.p_resig_cd1_high*sCD_buffer[i1,i2,i3][0]+(1-sCD_buffer[i1,i2,i3][0])*problem.p_resig_cd1_low
            p_tmp_CD2_resig =  problem.p_resig_cd2_high*sCD_buffer[i1,i2,i3][1]+(1-sCD_buffer[i1,i2,i3][1])*problem.p_resig_cd2_low


    p_tmp_FD_resig = problem.resig_fd

    p_tmp_CD1_add = problem.add_gw
    p_tmp_CD2_add = problem.add_od


    # Resignations - stochastic - state dependent - limit for proba
    n_FD_resig = np.random.binomial(state_t[0], p_tmp_FD_resig)

    if p_tmp_CD1_resig > 1:
        n_CD1_resig = np.random.binomial(state_t[1], 1)
    elif p_tmp_CD1_resig < 0:
        n_CD1_resig = np.random.binomial(state_t[1], 0)
    else:
        n_CD1_resig = np.random.binomial(state_t[1], p_tmp_CD1_resig)

    if p_tmp_CD2_resig > 1:
        n_CD2_resig = np.random.binomial(state_t[2], 1)
    elif p_tmp_CD2_resig < 0:
        n_CD2_resig = np.random.binomial(state_t[2], 0)
    else:
        n_CD2_resig = np.random.binomial(state_t[2], p_tmp_CD2_resig)


    if p_tmp_CD1_add > 1:
        n_CD1_add = np.random.binomial(np.round(state_t[1]),1)
    elif p_tmp_CD1_add <0:
        n_CD1_add = np.random.binomial(np.round(state_t[1]), 0)
    else:
        n_CD1_add = np.random.binomial(np.round(state_t[1]), p_tmp_CD1_add)

    if p_tmp_CD2_add > 1:
        n_CD2_add = np.random.binomial(np.round(state_t[2]),1)
    elif p_tmp_CD2_add < 0:
        n_CD2_add = np.random.binomial(np.round(state_t[2]),
                                       0)
    else:
        n_CD2_add = np.random.binomial(np.round(state_t[2]),
                                       p_tmp_CD2_add)

    state_t_prime[0] = max(0,state_t[0] - n_FD_resig)
    state_t_prime[1] = min(max(0,state_t[1] - n_CD1_resig + n_CD1_add),problem.maxCD1)
    state_t_prime[2] = min(max(0,state_t[2] - n_CD2_resig + n_CD2_add),problem.maxCD2)

    return state_t_prime