import nice.rascal_coefficients
from nice.nice_utilities import do_full_expansion
import tqdm 
import numpy as np

def get_rascal_coefficients_parallelized(p, structures, hypers, n_types,
                                         normalize = True, task_size = 100):
    
    tasks = []
    for i in range(0, len(structures), task_size):
        tasks.append([structures[i : i + task_size], hypers, n_types, normalize])
        
    def wrapped(task):
        return nice.rascal_coefficients.get_rascal_coefficients(*task)
    
    result = [res for res in tqdm.tqdm(p.imap(wrapped, tasks), total = len(tasks))]
    return np.concatenate(result, axis = 0)

'''def do_full_expansion_parallelized(p, clebsch, features, coefficients, l_max, task_size = 100):
    tasks = []
    for i in range(0, features.shape[0], task_size):
        tasks.append([clebsch, features[i : i + task_size], coefficients[i : i + task_size], l_max])
    def wrapped(task):
        return do_full_expansion(*task)
    
    same = []
    other = []
    for res in tqdm.tqdm(p.imap(wrapped, tasks), total = len(tasks)):
        same.append(res[0])
        other.append(res[1])
    
    return np.concatenate(same, axis = 0), np.concatenate(other, axis = 0)'''