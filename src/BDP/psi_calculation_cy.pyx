### File: psi_calculation_cy.py
### Author: Julius Luy, Gerhard Hiermann
### Date: November 10th 2023
### Description: This function implements the calculation of the value of being in a certain state based on the
### expected next states and rewards.

import cython
import scipy
import numpy as np
cimport numpy as np
from scipy.special cimport cython_special

from common.model_creation import model_creation

np.import_array()
ctypedef np.double_t DTYPE_double_t
ctypedef np.int_t DTYPE_int_t


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def calculate_psi_for(int fd, int cd1, int cd2, int t, double reward, problem: model_creation, double gamma, np.ndarray[DTYPE_double_t, ndim=4] vbar):


    cdef int num_CD1resign = cd1 + 1
    cdef int num_CD2resign = cd2 + 1
    cdef int num_CD1add = np.round(cd1)+1 # (problem.maxCD1 - cd1) + 1
    cdef int num_CD2add = np.round(cd2)+1 # (problem.maxCD2 - cd2) + 1

    cdef DTYPE_double_t p_tmp_CD1_resign = problem.p_resig_cd1[fd,cd1,cd2,t]
    cdef DTYPE_double_t p_tmp_CD2_resign = problem.p_resig_cd2[fd,cd1,cd2,t]

    cdef DTYPE_double_t p_tmp_CD1_add = problem.p_add_cd1[fd,cd1,cd2,t]
    cdef DTYPE_double_t p_tmp_CD2_add = problem.p_add_cd2[fd,cd1,cd2,t]

    cdef int i
    cdef np.ndarray[DTYPE_double_t] pCD1resign = np.empty(num_CD1resign, dtype=np.double)
    for i in range(0, num_CD1resign):
        pCD1resign[i] = cython_special.binom(cd1, i) * p_tmp_CD1_resign ** i * (1 - p_tmp_CD1_resign) ** (cd1 - i)

    cdef np.ndarray[DTYPE_double_t] pCD2resign = np.empty(num_CD2resign, dtype=np.double)
    for i in range(0, num_CD2resign):
        pCD2resign[i] = cython_special.binom(cd2, i) * p_tmp_CD2_resign ** i * (1 - p_tmp_CD2_resign) ** (cd2 - i)

    cdef np.ndarray[DTYPE_double_t] pCD1add = np.empty(num_CD1add, dtype=np.double)
    for i in range(0, num_CD1add):
        pCD1add[i] = cython_special.binom(num_CD1add-1, i) * p_tmp_CD1_add ** i * (1 - p_tmp_CD1_add) ** ((num_CD1add-1) - i)

    cdef np.ndarray[DTYPE_double_t] pCD2add = np.empty(num_CD2add, dtype=np.double)
    for i in range(0, num_CD2add):
        pCD2add[i] = cython_special.binom(num_CD2add-1, i) * p_tmp_CD2_add ** i * (1 - p_tmp_CD2_add) ** ((num_CD2add-1) - i)

    cdef int t_next_state
    if problem.tmax == 0:
        t_next_state = t
    else:
        t_next_state = t+1

    cdef double epsilon
    if num_CD1add > 1 and num_CD2add > 1:
        epsilon = 0#  0.01/(num_CD1add*num_CD2add)
    else:
        epsilon = 0 # 0.01/4

    cdef double psi = calculate_psi_loop(fd, cd1, cd2, gamma, pCD1add, pCD1resign,
                              pCD2add, pCD2resign, reward, problem.maxCD1, problem.maxCD2, t_next_state, vbar,epsilon)

    return psi


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def calculate_psi_loop(int i, int j, int k,
                       double gamma,
                       np.ndarray[DTYPE_double_t] pCD1add, np.ndarray[DTYPE_double_t] pCD1resign,
                       np.ndarray[DTYPE_double_t] pCD2add, np.ndarray[DTYPE_double_t] pCD2resign,
                       DTYPE_double_t reward,
                       int maxCD1,
                       int maxCD2,
                       int t_next_state, np.ndarray[DTYPE_double_t, ndim=4] vbar,double epsilon):

    cdef double eps = epsilon

    cdef int len_pCD1resign = pCD1resign.shape[0]
    cdef int len_pCD1add = pCD1add.shape[0]

    cdef int len_pCD2resign = pCD2resign.shape[0]
    cdef int len_pCD2add = pCD2add.shape[0]
    cdef np.ndarray[DTYPE_double_t] pCD2outcome = np.empty(len_pCD2resign * len_pCD2add, dtype=np.double)
    cdef np.ndarray[DTYPE_int_t] CD2outcome = np.empty(len_pCD2resign * len_pCD2add, dtype=np.int)

    cdef DTYPE_double_t pOutcome

    cdef int cnt = 0
    cdef int l
    cdef int v
    for l in range(0, len_pCD2resign):
        for v in range(0, len_pCD2add):
            pOutcome = pCD2resign[l] * pCD2add[v]
            if pOutcome >= eps:
                pCD2outcome[cnt] = pOutcome
                CD2outcome[cnt] = -l + v
                cnt += 1

    cdef int len_CD2outcome = cnt

    cdef DTYPE_double_t psi = reward

    cdef int x = i
    cdef int y
    cdef int z

    cdef int l1
    cdef int v1
    for l1 in range(0, len_pCD1resign):
        for v1 in range(0, len_pCD1add):
            pOutcome = pCD1resign[l1] * pCD1add[v1]
            if pOutcome >= eps:
                y = max(min(j - l1 + v1, maxCD1),0)
                for v in range(0, len_CD2outcome):
                    z = max(min(k + CD2outcome[v], maxCD2),0)
                    psi += pOutcome * pCD2outcome[v] * gamma * vbar[t_next_state, x, y, z]  # Increment psi by the current combination's expected reward # calculate new values



    return psi
