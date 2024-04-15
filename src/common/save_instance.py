import os
from datetime import datetime
import numpy as np
def save_instance(vbar,act,results_path):
    values_folder = results_path.joinpath('vbar.npy')
    f = open(values_folder,'wb')
    np.save(f,vbar)
    actions_folder = results_path.joinpath('act.npy')
    f = open(actions_folder,'wb')
    np.save(f,act)




