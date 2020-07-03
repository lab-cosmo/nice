from scipy.stats import ortho_group
from scipy.linalg import det
import numpy as np

def rotate_env(positions):
    A = ortho_group.rvs(3)
    if (det(A) < 0):
        A = -A
    
    return np.sum(A[np.newaxis, :, :] * positions[:, np.newaxis, :], axis = -1)
    