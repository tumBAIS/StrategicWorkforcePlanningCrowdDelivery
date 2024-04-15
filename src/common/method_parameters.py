"""
File: method_parameters.py
Author: Julius Luy
Date: November 10th 2023
Description: This class implements and stores all methodological settings and hyperparameters.

Parameters
----------
problem: object
    Class that implements the MDP's environment.
base_configuration : object
    Contains all configuration parameters.
"""

import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from csv import writer


class method_parameters:

    def __init__(self,problem,base_configuration):
        input_file = base_configuration.analysisPath / base_configuration.input_file
        x = {}
        with open(input_file, 'r',encoding="utf-8") as fd:
            reader = csv.reader(fd)
            for row in reader:
                x[row[0]] = row[1]
        fd.close()

        self.gamma = float(x['gamma'])

        self.max_epochs = int(x['max_epochs'])
        self.alpha = float(x['alpha'])
        self.N_testing = int(x['n_testing'])
        if problem.firing_decisions == 'yes':
            self.fd_discretization = int(x['fd_discretization'])
        else:
            self.fd_discretization = 1
        self.cd_discretization = int(x['cd_discretization'])
        self.discretization_technique = x['discretization_technique']

        problem.action_space = np.arange(0,problem.maxFD+self.fd_discretization,self.fd_discretization)

        # Obtain mapping between number of FDs, GWs, ODs, given a certain level of aggregation
        self.fd_vec_aggr,self.fd_vec_disc,self.gw_vec,self.od_vec = self.discretization_points(problem,self.fd_discretization,self.cd_discretization, self.discretization_technique)
        self.state_counter = {}
        self.v_counter = {}

    def findDiscVector(self,maxCD, cd_discretization, id_):
        if id_ == maxCD:
            vec = np.floor(np.arange(0, maxCD + 1) / cd_discretization)
        elif id_ == 0:
            vec = np.zeros(maxCD + 1)
        else:
            tmp0 = np.floor(np.arange(0, id_ + 1) / cd_discretization)
            tmp1 = (np.max(tmp0) + 1) * np.ones(maxCD + 1 - (id_ + 1))
            vec = np.array([*tmp0, *tmp1])
        return vec

    def discretization_points(self,problem, fd_discretization, cd_discretization, discretization_technique):
        # Evaluation step size for GWs
        gw_vec = np.zeros((problem.maxCD1 + 1, problem.tmax + 1))
        od_vec = np.zeros((problem.maxCD2 + 1, problem.tmax + 1))

        fd_vec_aggr = np.zeros((problem.maxFD + 1, problem.tmax + 1))
        fd_vec_disc = np.zeros((problem.maxFD + 1, problem.tmax + 1))
        if problem.firing_decisions == 'yes':
            for t in range(0, problem.tmax + 1):
                fd_vec_aggr[:, t] = np.floor(np.arange(0, problem.maxFD + 1) / fd_discretization)
            fd_vec_disc[:, :] = fd_vec_aggr[:, :]
        else:
            for t in range(0, problem.tmax + 1):
                fd_vec_aggr[:, t] = np.zeros(problem.maxFD + 1)
                fd_vec_disc[:, t] = np.arange(0, problem.maxFD + 1)

        for t in range(0, problem.tmax + 1):

            if discretization_technique == 'homogeneous':
                for t in range(0, problem.tmax + 1):
                    gw_vec[:, t] = np.floor(np.arange(0, problem.maxCD1 + 1) / cd_discretization)
                    od_vec[:, t] = np.floor(np.arange(0, problem.maxCD2 + 1) / cd_discretization)
            elif discretization_technique == 'inhomogeneous':
                n = int(0.5 * problem.maxFD)

                gw_maxima = []
                n_fd = np.zeros(n + 1, dtype=int)
                for i in range(0, n + 1):
                    n_fd[i] = int(np.round(i * problem.maxFD / n))
                    gw_maxima.append(np.argmax(
                        self.findDiscVector(problem.maxCD1, cd_discretization, np.argmax(problem.rew[n_fd[i], :, 0, t]))))
                gw_maxima = np.array(gw_maxima)
                gw_maximum_avg = np.average(gw_maxima)
                gw_maximum_avg = int(gw_maximum_avg)
                gw_vec_tmp1 = np.floor(np.arange(0, gw_maximum_avg + 1) / cd_discretization)
                gw_vec_tmp2 = (np.max(gw_vec_tmp1) + 1) * np.ones(problem.maxCD1 + 1 - (gw_maximum_avg + 1))
                gw_vec[:, t] = np.array([*gw_vec_tmp1, *gw_vec_tmp2])

                od_maxima = []
                n_fd = np.zeros(n + 1, dtype=int)
                for i in range(0, n + 1):
                    n_fd[i] = int(np.round(i * problem.maxFD / n))
                    od_maxima.append(np.argmax(
                        self.findDiscVector(problem.maxCD2, cd_discretization, np.argmax(problem.rew[n_fd[i], 0, :, t]))))
                od_maxima = np.array(od_maxima)
                od_maximum_avg = np.average(od_maxima)
                od_maximum_avg = int(od_maximum_avg)
                od_vec_tmp1 = np.floor(np.arange(0, od_maximum_avg + 1) / cd_discretization)
                od_vec_tmp2 = (np.max(od_vec_tmp1) + 1) * np.ones(problem.maxCD2 + 1 - (od_maximum_avg + 1))
                od_vec[:, t] = np.array([*od_vec_tmp1, *od_vec_tmp2])

        return fd_vec_aggr, fd_vec_disc, gw_vec, od_vec
