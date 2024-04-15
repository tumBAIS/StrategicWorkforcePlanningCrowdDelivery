import numpy as np
def revenue_calculation(problem):
    rev = np.zeros((problem.loc,problem.loc,3))
    rev[:,:,0] = problem.avg_rev_per_request
    rev[:, :, 1] = problem.avg_rev_per_request
    rev[:, :, 2] = problem.avg_rev_per_request
    return rev