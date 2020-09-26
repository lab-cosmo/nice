import numpy as np
from nice.nice_utilities import Data


class ParityDefinitionChanger():
    def _init__(self):
        self.fitted_ = True

    def is_fitted(self):
        return self.fitted_

    def transform(self, first_data, second_data):
        l_max = first_data.covariants_.shape[2] - 1
        new_first_sizes, new_second_sizes = [], []
        for lambd in range(l_max + 1):
            if (lambd % 2 == 0):
                new_first_sizes.append(first_data.actual_sizes_[lambd])
                new_second_sizes.append(second_data.actual_sizes_[lambd])
            else:
                new_first_sizes.append(second_data.actual_sizes_[lambd])
                new_second_sizes.append(first_data.actual_sizes_[lambd])

        new_first_sizes, new_second_sizes = np.array(new_first_sizes,
                                                     dtype=np.int32), np.array(
                                                         new_second_sizes,
                                                         dtype=np.int32)
        new_first_shape = list(first_data.covariants_.shape)
        new_first_shape[1] = np.max(new_first_sizes)

        new_second_shape = list(second_data.covariants_.shape)
        new_second_shape[1] = np.max(new_second_sizes)

        new_first_covariants = np.empty(new_first_shape)
        new_second_covariants = np.empty(new_second_shape)

        for lambd in range(l_max + 1):  # todo may be do copying in parallel
            if (lambd % 2 == 0):
                new_first_covariants[:, :new_first_sizes[lambd], lambd, :(
                    2 * lambd +
                    1)] = first_data.covariants_[:, :new_first_sizes[lambd],
                                                 lambd, :(2 * lambd + 1)]
                new_second_covariants[:, :new_second_sizes[lambd], lambd, :(
                    2 * lambd +
                    1)] = second_data.covariants_[:, :new_second_sizes[lambd],
                                                  lambd, :(2 * lambd + 1)]
            else:
                new_first_covariants[:, :new_first_sizes[lambd], lambd, :(
                    2 * lambd +
                    1)] = second_data.covariants_[:, :new_first_sizes[lambd],
                                                  lambd, :(2 * lambd + 1)]
                new_second_covariants[:, :new_second_sizes[lambd], lambd, :(
                    2 * lambd +
                    1)] = first_data.covariants_[:, :new_second_sizes[lambd],
                                                 lambd, :(2 * lambd + 1)]

        if (first_data.importances_ is None) or (second_data.importances_ is
                                                 None):
            new_first_importances = None
            new_second_importances = None
        else:
            new_first_importances = np.empty(
                [np.max(new_first_sizes), l_max + 1])
            new_second_importances = np.empty(
                [np.max(new_second_sizes), l_max + 1])

            for lambd in range(l_max + 1):
                if (lambd % 2 == 0):
                    new_first_importances[:new_first_sizes[
                        lambd], lambd] = first_data.importances_[:
                                                                 new_first_sizes[
                                                                     lambd],
                                                                 lambd]
                    new_second_importances[:new_second_sizes[
                        lambd], lambd] = second_data.importances_[:
                                                                  new_second_sizes[
                                                                      lambd],
                                                                  lambd]
                else:
                    new_first_importances[:new_first_sizes[
                        lambd], lambd] = second_data.importances_[:
                                                                  new_first_sizes[
                                                                      lambd],
                                                                  lambd]
                    new_second_importances[:new_second_sizes[
                        lambd], lambd] = first_data.importances_[:
                                                                 new_second_sizes[
                                                                     lambd],
                                                                 lambd]

        return Data(new_first_covariants, new_first_sizes, new_first_importances), \
               Data(new_second_covariants, new_second_sizes, new_second_importances)


class InitialScaler():
    def __init__(self):
        self.fitted_ = False

    def fit(self, coefficients):
        total = 0.0
        total_values = 0

        for l in range(coefficients.shape[2]):
            total += np.sum((coefficients[:, :, l, 0:(2 * l + 1)])**2)
            total_values += coefficients.shape[0] * coefficients.shape[1] * (
                2 * l + 1)

        average = total / total_values
        self.multiplier_ = 1.0 / np.sqrt(average)
        self.fitted_ = True

    def transform(self, coefficients):
        if (not self.fitted_):
            raise NotFittedError(
                "instance of {} is not fitted. It can not transform anything".
                format(type(self).__name__))

        return coefficients * self.multiplier_

    def is_fitted(self):
        return self.fitted_


class InitialTransformer():
    def __init__(self):
        self.fitted_ = True

    def transform(self, coefficients):
        l_max = coefficients.shape[2] - 1
        even_coefficients = np.copy(coefficients)
        even_coefficients_sizes = [
            coefficients.shape[1] if i % 2 == 0 else 0
            for i in range(l_max + 1)
        ]

        odd_coefficients = np.copy(coefficients)
        odd_coefficients_sizes = [
            coefficients.shape[1] if i % 2 == 1 else 0
            for i in range(l_max + 1)
        ]

        return Data(even_coefficients,
                    even_coefficients_sizes), Data(odd_coefficients,
                                                   odd_coefficients_sizes)

    def is_fitted(self):
        return self.fitted_
