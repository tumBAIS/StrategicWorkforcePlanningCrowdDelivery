"""
File: psi_calculation.py
Author: Julius Luy
Date: November 10th 2023
Description: This function implements returns the value of being in a certain state based on the
expected next states and rewards. If the state (fd,cd1,cd2) is cached it takes the value from a cached vector. If not
tt calls a cython function which efficiently calculates the value of being in a certain state based on the
expected next states and rewards.

Parameters
----------
fd : int
     No. of fds.
cd1,cd2 : int
    No. of crowdsourced drivers.
t : int
    Time step.
reward : float
    Total costs.
problem: object
    Class that implements the MDP's environment.
gamma: float
    Discount factor
vbar: float
    State value
vec_buffer: float dict
    Caches state values for different driver states

Returns
-------
psi: State value of (fd,cd1,cd2) in t.
"""

from typing import Dict

from common.model_creation import model_creation
from BDP.psi_calculation_cy import calculate_psi_for as calculate_psi_for_uncached


def calculate_psi_for(fd: int, cd1: int, cd2: int, t: int, reward: float, problem: model_creation, gamma: float,
                      vbar, vec_buffer: Dict[str, float]) -> float:
    string = str(fd)+' '+str(cd1)+' '+str(cd2)
    if string in vec_buffer:
        psi = vec_buffer[string]
    else:
        psi = calculate_psi_for_uncached(fd, cd1, cd2, t, reward, problem, gamma, vbar)
        vec_buffer[string] = psi
    return psi
