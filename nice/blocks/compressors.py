import numpy as np
from nice.unrolling_individual_pca import UnrollingIndividualPCA

# from cython.parallel cimport prange

from nice.thresholding import get_thresholded_tasks
from nice.nice_utilities import do_partial_expansion, Data, get_sizes
from nice.clebsch_gordan import ClebschGordan, check_clebsch_gordan
from nice.packing import unite_parallel, subtract_parallel
from nice.packing import pack_dense, unpack_dense
from parse import parse
import warnings
from sklearn.linear_model import Ridge
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.decomposition import PCA


def get_num_fit(desired_num, block_size):
    if desired_num % block_size == 0:
        return desired_num // block_size
    else:
        return (desired_num // block_size) + 1


class IndividualLambdaPCAs:
    ''' Block to do pca step for covariants of single parity. It operates with instances of Data class'''
    def __init__(self, n_components=None, num_to_fit="10x"):
        self.n_components_ = n_components
        self.num_to_fit_ = num_to_fit
        self.fitted_ = False

    def get_importances(self):
        if not self.fitted_:
            raise NotFittedError(
                ("instance of {} is not fitted. "
                 "Thus importances are not available.").format(
                     type(self).__name__))
        result = np.empty([self.max_n_components_, self.l_max_ + 1])
        for lambd in range(self.l_max_ + 1):
            if self.pcas_[lambd] is not None:
                result[:self.pcas_[lambd].n_components,
                       lambd] = self.pcas_[lambd].importances_

        actual_sizes = []
        for lambd in range(self.l_max_ + 1):
            if self.pcas_[lambd] is not None:
                actual_sizes.append(self.pcas_[lambd].n_components)
            else:
                actual_sizes.append(0)
        return result

    def fit(self, data):

        self.l_max_ = data.covariants_.shape[2] - 1
        self.pcas_ = []
        self.reduction_happened_ = False
        self.max_n_components_ = -1
        for lambd in range(self.l_max_ + 1):
            if data.actual_sizes_[lambd] > 0:
                if self.n_components_ is None:
                    n_components_now = data.actual_sizes_[lambd]
                else:
                    n_components_now = self.n_components_

                self.max_n_components_ = max(self.max_n_components_,
                                             n_components_now)

                if data.covariants_.shape[0] * (lambd + 1) < n_components_now:
                    raise ValueError((
                        "not enough data to fit pca, number of vectors is {}, "
                        "dimensionality of single vector (lambd + 1) is {}, "
                        "i. e. total number of points is {}, "
                        "while number of components is {}.").format(
                            data.covariants_.shape[0],
                            lambd + 1,
                            data.covariants_.shape[0] * (lambd + 1),
                            n_components_now,
                        ))

                if type(self.num_to_fit_) is str:
                    multiplier = int(parse("{}x", self.num_to_fit_)[0])
                    num_fit_now = get_num_fit(multiplier * n_components_now,
                                              (lambd + 1))
                else:
                    num_fit_now = self.num_to_fit_
                    if num_fit_now * (lambd + 1) < n_components_now:
                        raise ValueError(
                            ("specified parameter num fit ({}) is too "
                             "small to fit pca with number of components {}."
                             ).format(num_fit_now, n_components_now))

                if data.covariants_.shape[0] * (lambd + 1) < num_fit_now:
                    warnings.warn(
                        ("given data is less than desired number "
                         "of points to fit pca. "
                         "Desired number of points to fit pca is {}, "
                         "while number of vectors is {}, "
                         "dimensionality of single vector (lambd + 1) is {}, "
                         "i. e. total number of points is {}. "
                         "Number of pca components is {}.").format(
                             num_fit_now,
                             data.covariants_.shape[0],
                             (lambd + 1),
                             data.covariants_.shape[0] * (lambd + 1),
                             n_components_now,
                         ),
                        RuntimeWarning,
                    )

                if n_components_now < data.actual_sizes_[lambd]:
                    self.reduction_happened_ = True
                pca = UnrollingIndividualPCA(n_components=n_components_now)
                pca.fit(
                    data.covariants_[:num_fit_now, :data.actual_sizes_[lambd],
                                     lambd, :],
                    lambd,
                )
                self.pcas_.append(pca)
            else:
                self.pcas_.append(None)
        self.fitted_ = True
        self.importances_ = self.get_importances()

    def transform(self, data):
        if not self.fitted_:
            raise NotFittedError(
                ("instance of {} is not fitted. "
                 "It can not transform anything.").format(type(self).__name__))
        result = np.empty([
            data.covariants_.shape[0],
            self.max_n_components_,
            self.l_max_ + 1,
            2 * self.l_max_ + 1,
        ])
        new_actual_sizes = np.zeros([self.l_max_ + 1], dtype=np.int32)
        for lambd in range(self.l_max_ + 1):
            if self.pcas_[lambd] is not None:
                now = self.pcas_[lambd].transform(
                    data.covariants_[:, :data.actual_sizes_[lambd], lambd, :],
                    lambd)
                result[:, :now.shape[1], lambd, :(2 * lambd + 1)] = now
                new_actual_sizes[lambd] = now.shape[1]
            else:
                new_actual_sizes[lambd] = 0

        return Data(result, new_actual_sizes, importances=self.importances_)

    def is_fitted(self):
        return self.fitted_


class IndividualLambdaPCAsBoth:
    ''' Block to do pca step for covariants of both parities. It operates with even-odd pairs of instances of Data class'''
    def __init__(self, *args, **kwargs):
        self.even_pca_ = IndividualLambdaPCAs(*args, **kwargs)
        self.odd_pca_ = IndividualLambdaPCAs(*args, **kwargs)
        self.fitted_ = False

    def fit(self, data_even, data_odd):

        self.even_pca_.fit(data_even)
        self.odd_pca_.fit(data_odd)
        self.fitted_ = True

    def transform(self, data_even, data_odd):
        if not self.fitted_:
            raise NotFittedError(
                ("instance of {} is not fitted. "
                 "It can not transform anything.").format(type(self).__name__))
        return self.even_pca_.transform(data_even), self.odd_pca_.transform(
            data_odd)

    def is_fitted(self):
        return self.fitted_


class InvariantsPCA(PCA):
    ''' Block to do pca step for invariants. It operates with 2d numpy arrays'''
    def __init__(self, *args, num_to_fit="10x", **kwargs):
        self.num_to_fit_ = num_to_fit
        self.fitted_ = False
        return super().__init__(*args, **kwargs)

    def _my_representation(self):
        if (self.fitted_):
            return "Instance of InvariantsPCA, fitted"
        else:
            return "Instance of InvariantsPCA, not fitted"

    def __repr__(self):
        return self._my_representation()

    def __str__(self):
        return self._my_representation()

    def process_input(self, X):
        if (self.n_components is None):
            self.n_components = X.shape[1]
        if (self.n_components > X.shape[1]):
            self.n_components = X.shape[1]

        if type(self.num_to_fit_) is str:
            multiplier = int(parse("{}x", self.num_to_fit_)[0])
            num_fit_now = multiplier * self.n_components
        else:
            num_fit_now = self.num_to_fit_

        if self.n_components > X.shape[0]:
            raise ValueError(
                ("not enough data to fit pca. "
                 "Number of environments is {}, number of components is {}."
                 ).format(X.shape[0], self.n_components))

        if num_fit_now > X.shape[0]:
            warnings.warn(("Amount of provided data is less "
                           "than the desired one to fit PCA. "
                           "Number of components is {}, "
                           "desired number of environments is {}, "
                           "actual number of environments is {}.").format(
                               self.n_components, num_fit_now, X.shape[0]))

        return X[:num_fit_now]

    def fit(self, X):

        res = super().fit(self.process_input(X))
        self.fitted_ = True
        return res

    def fit_transform(self, X):
        res = super().fit_transform(self.process_input(X))
        self.fitted_ = True
        return res

    def transform(self, X):
        if not self.fitted_:
            raise NotFittedError(
                ("instance of {} is not fitted. "
                 "It can not transform anything.").format(type(self).__name__))
        return super().transform(X)

    def is_fitted(self):
        return self.fitted_
