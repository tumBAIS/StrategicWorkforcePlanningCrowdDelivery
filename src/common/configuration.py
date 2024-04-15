"""
File: configuration.py
Author: Julius Luy
Date: November 10th 2023
Description: This class stores all configuration parameters read from the input .csv file.

Initialization parameters
----------
cmd_params : string
    Contains the location of the inoput .csv
mode : string
    Indicates whether the algorithm should be called in training or test mode.
"""


# Import of Python packages
import numpy as np
import csv
import sys
import os

class configuration:
    def __init__(self,cmd_params,mode):
        x = {}
        print(cmd_params)
        with open(cmd_params, 'r',encoding="utf-8") as fd:
            reader = csv.reader(fd)
            for row in reader:
                x[row[0]] = row[1]
        fd.close()

        self.analysisPath = cmd_params.parent
        self.input_file = cmd_params
        self.results_path = cmd_params.parent.parent / 'results' / x['instance_name']
        if not self.results_path.exists() and mode != 'test':
            os.mkdir(self.results_path)
        self.include_firing_decisions = str(x['include_firing_decisions'])
        self.solution_approach = x['solution_approach']
        self.init_fd = int(float(x['init_fd']))
        self.init_gw = int(float(x['init_gw']))
        self.init_od = int(float(x['init_od']))
        self.init_state = np.array([self.init_fd, self.init_gw, self.init_od])
        self.reward_calculation = str(x['rew'])
        self.load_existing_costs = str(x['load_existing_costs'])
        self.load_data = str(x['load_data'])
        self.save_data = str(x['save_data'])
        self.seed = int(x['seed'])
        self.mode = mode
        self.n_check_policy = int(x['n_check_policy'])
        self.n_convergence_check = int(x['n_convergence_check'])
        try:
            self.load_existing_slopes = x['load_existing_slopes']
        except:
            self.load_existing_slopes = 'no'
