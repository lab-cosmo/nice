import nice.rascal_coefficients
#from nice.nice_utilities import do_full_expansion
import tqdm
import numpy as np


def get_rascal_coefficients_parallelized(p,
                                         structures,
                                         hypers,
                                         n_types,
                                         normalize=True,
                                         task_size=100):

    tasks = []
    for i in range(0, len(structures), task_size):
        tasks.append([structures[i:i + task_size], hypers, n_types, normalize])

    def wrapped(task):
        return nice.rascal_coefficients.get_rascal_coefficients(*task)

    result = [
        res for res in tqdm.tqdm(p.imap(wrapped, tasks), total=len(tasks))
    ]
    return np.concatenate(result, axis=0)
