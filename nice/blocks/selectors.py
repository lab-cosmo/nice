import numpy as np
from nice.packing import pack_dense, unpack_dense


class InvariantsVarianceSelector:
    def __init__(self, n_components):
        self.n_components_ = n_components

    def get_subselection(self, invariants, previous):
        if (invariants.shape[1] <= self.n_components_):
            return np.ones([invariants.shape[1]], dtype = np.bool), np.ones([invariants.shape[1]])

        variances = np.mean(invariants * invariants, axis = 0)
        indices = np.argsort(variances)[::-1]
        mask = np.zeros([invariants.shape[1]], dtype = np.bool)
        mask[indices[:self.n_components_]] = True
        return [mask, np.ones([self.n_components_])]


#for comparing purposes
class InvariantsRandomSelector:
    def __init__(self, n_components):
        self.n_components_ = n_components

    def get_subselection(self, invariants, previous):
        if (invariants.shape[1] <= self.n_components_):
            return np.ones([invariants.shape[1]], dtype=np.bool), np.ones([invariants.shape[1]])

        indices = np.random.permutation(invariants.shape[1])
        mask = np.zeros([invariants.shape[1]], dtype=np.bool)
        mask[indices[:self.n_components_]] = True
        return [mask, np.ones([self.n_components_])]


class IndividualSelector:
    def __init__(self, base):
        self.base_ = base

    def get_subselection(self, covariants, l, previous):
        covariants_p = pack_dense(covariants, l, covariants.shape[1], covariants.shape[1])
        previous_p = [pack_dense(el, l, el.shape[1], el.shape[1]) for el in previous]
        return self.base_.get_subselection(covariants_p, previous_p)


class CovariantsSelector:
    def __init__(self, base):
        self.base_ = base

    def get_subselection(self, data, previous):
        ans = []
        for lambd in range(data.covariants_.shape[2]):
            if (data.actual_sizes_[lambd] > 0):
                previous_now = [el.covariants_[:, :el.actual_sizes_[lambd], lambd, :] for el in previous
                                if el.actual_sizes_[lambd] > 0]

                mask, multipliers = IndividualSelector(self.base_).get_subselection(data.covariants_[:, :data.actual_sizes_[lambd], lambd, :],
                                                              lambd, previous_now)
                ans.append([mask, multipliers])
            else:
                ans.append([np.array([], dtype = bool), np.array([])])

        return ans

class CovariantsSelectorBoth:
    def __init__(self, base):
        self.base_ = base

    def get_subselection(self, even, even_previous, odd, odd_previous):
        return CovariantsSelector(self.base_).get_subselection(even, even_previous),\
               CovariantsSelector(self.base_).get_subselection(odd, odd_previous)

'''class InvariantsFPSSelector:
    def __init__(self, n_components):
        self.n_components_ = n_components

    def get_subselection(self, invariants, previous):
        if (invariants.shape[1] <= self.n_components_):
            return np.ones([invariants.shape[1]], dtype=np.bool), np.ones([invariants.shape[1]])

        previous = np.concatenate(previous, axis = 1)'''
