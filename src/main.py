"""
File: main.py
Author: Julius Luy
Date: November 10th 2023
Description: This function triggers the code execution of the paper.

Parameters
----------
sys.argv[1] : string
    Location of input .csv file
sys.argv[2] : string
    Indicating whether training (train) or test (test) mode
"""

# Import of Python packages
import numpy as np
import csv
import shutil
import os
import sys
from pathlib import Path
current_working_dir = Path(os.getcwd())
main_project_dir = str(current_working_dir)
sys.path.append(main_project_dir+'/cpp/cmake-build-release/lib')

# Import of own functions/classes
from common.model_creation import model_creation
from common.method_parameters import method_parameters
from BDP.backward_DP import backward_DP
from common.configuration import configuration
from PVFA.piecewise_linear_approx import PLVFA
from common.csv_operations import readCsvFile,modCsvFile


if __name__ == '__main__':

    if len(sys.argv) > 1:
        cmd_params = Path('.').cwd().parent / 'data' / str(sys.argv[1])
        mode = str(sys.argv[2])
    else:
        mode = 'train'
        if Path('.').cwd().name == 'src':
            src_path = Path('.').cwd().parent
            input_files = os.listdir(src_path / 'data')
        else:
            src_path = Path('.').cwd()
            input_files = os.listdir(src_path / 'data')

        if len(input_files) == 0:
            print('No input file given - abort!')
            os.abort()
        else:
            for file in input_files:
                if file[-4:] == '.csv':
                    cmd_params = src_path / 'data' / file

    base_configuration = configuration(cmd_params,mode)

    np.random.seed(base_configuration.seed)

    analysisPath = base_configuration.analysisPath
    problem = model_creation(base_configuration)
    method = method_parameters(problem,base_configuration)

    if base_configuration.solution_approach == 'exact':
        backward_DP(problem, method,base_configuration.results_path)
    else:
        pvfa_agent = PLVFA(problem,method,base_configuration,False)
        v,total_costs_hist = pvfa_agent.piecewise_linear_approx()












