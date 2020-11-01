from nice.thresholding import get_thresholded_tasks
from nice.nice_utilities import do_partial_expansion, Data, get_sizes
from nice.clebsch_gordan import ClebschGordan, check_clebsch_gordan
import numpy as np
from sklearn.exceptions import NotFittedError


def amplitudes_criterion(data):
    return data.get_amplitudes()


def standardized_amplitudes_criterion(data):
    amplitudes = data.get_amplitudes()
    for lambd in range(amplitudes.shape[1]):
        amplitudes[:data.actual_sizes_[lambd], lambd] /= np.sqrt(2 * lambd + 1)
    return amplitudes

class ThresholdExpansioner:
    ''' Block to do Clebsch-Gordan iteration. It uses two even-odd pairs of Data instances with covariants
    to produce new ones. If first even-odd pair contains covariants of body order v1, and the second v2, body
    order of the result would be v1 + v2. '''
    def __init__(self, num_expand=None, mode="covariants", criterion = "standardized_amplitudes", num_threads=None):
        if num_expand is None:
            self.num_expand_ = -1
        else:
            self.num_expand_ = num_expand
        if criterion == "amplitudes":
            self.criterion_ = amplitudes_criterion
        else:
            if criterion == "standardized_amplitudes":
                self.criterion_ = standardized_amplitudes_criterion

        self.mode_ = mode
        self.num_threads_ = num_threads
        self.fitted_ = False

    def apply_subselection(self, even_subselection, odd_subselection):
        if not self.fitted_:
            raise NotFittedError(
                "Expansioner must be fitted before applying subselection".
                format(type(self).__name__))

        if self.mode_ == "covariants":
            even_mask = np.zeros([self.task_even_even_[0].shape[0] + self.task_odd_odd_[0].shape[0]], dtype=bool)
            odd_mask = np.zeros([self.task_even_odd_[0].shape[0] + self.task_odd_even_[0].shape[0]], dtype=bool)

            even_multipliers, odd_multipliers = [], []
            for lambd in range(self.l_max_ + 1):
                even_positions = np.concatenate([(self.task_even_even_[0][:, 4] == lambd),
                                                 (self.task_odd_odd_[0][:, 4] == lambd)], axis = 0)

                odd_positions = np.concatenate([(self.task_even_odd_[0][:, 4] == lambd),
                                                         (self.task_odd_even_[0][:, 4] == lambd)], axis=0)

                even_mask[even_positions] = even_subselection[lambd][0]

                odd_mask[odd_positions] = odd_subselection[lambd][0]

                even_multipliers.append(even_subselection[lambd][1])
                odd_multipliers.append(odd_subselection[lambd][1])

        else:
            even_mask = even_subselection[0]
            odd_mask = odd_subselection[0]
            even_multipliers = even_subselection[1]
            odd_multipliers = odd_subselection[1]

        #print(even_mask.shape, self.task_even_even_[0].shape, self.task_odd_odd_[0].shape, self.task_even_even_[0].shape[0], even_mask[self.task_even_even_[0].shape[0]:].shape)
        #print(self.task_even_even_[0].shape)
        #print(self.task_odd_odd_[0].shape)
        first_submask = even_mask[:self.task_even_even_[0].shape[0]]
        second_submask = even_mask[self.task_even_even_[0].shape[0]:]
        self.task_even_even_[0] = self.task_even_even_[0][first_submask]
        #print(self.task_odd_odd_[0].shape, submask.shape)
        self.task_odd_odd_[0] = self.task_odd_odd_[0][second_submask]

        first_submask = odd_mask[:self.task_even_odd_[0].shape[0]]
        second_submask = odd_mask[self.task_even_odd_[0].shape[0]:]
        self.task_even_odd_[0] = self.task_even_odd_[0][first_submask]
        self.task_odd_even_[0] = self.task_odd_even_[0][second_submask]

        #print(self.task_even_even_[0].shape)
        #print(self.task_odd_odd_[0].shape)
        self.new_even_size_ = np.max(
            get_sizes(self.l_max_, self.task_even_even_[0], self.mode_) +
            get_sizes(self.l_max_, self.task_odd_odd_[0], self.mode_))

        self.new_odd_size_ = np.max(
            get_sizes(self.l_max_, self.task_even_odd_[0], self.mode_) +
            get_sizes(self.l_max_, self.task_odd_even_[0], self.mode_))

        self.even_multipliers_ = even_multipliers
        self.odd_multipliers_ = odd_multipliers


    def fit(self,
            first_even,
            first_odd,
            second_even,
            second_odd,
            clebsch_gordan=None):

        self.l_max_ = first_even.covariants_.shape[2] - 1

        (
            self.task_even_even_,
            self.task_odd_odd_,
            self.task_even_odd_,
            self.task_odd_even_,
        ) = get_thresholded_tasks(
            self.criterion_,
            first_even,
            first_odd,
            second_even,
            second_odd,
            self.num_expand_,
            self.l_max_,
            self.mode_,
        )

        if clebsch_gordan is None:
            self.clebsch_ = ClebschGordan(self.l_max_)
        else:
            check_clebsch_gordan(clebsch_gordan, self.l_max_)
            self.clebsch_ = clebsch_gordan

        self.new_even_size_ = np.max(
            get_sizes(self.l_max_, self.task_even_even_[0], self.mode_) +
            get_sizes(self.l_max_, self.task_odd_odd_[0], self.mode_))

        self.new_odd_size_ = np.max(
            get_sizes(self.l_max_, self.task_even_odd_[0], self.mode_) +
            get_sizes(self.l_max_, self.task_odd_even_[0], self.mode_))

        self.even_multipliers_ = None
        self.odd_multipliers_ = None

        self.fitted_ = True

    def transform(self, first_even, first_odd, second_even, second_odd):
        if not self.fitted_:
            raise NotFittedError(
                "instance of {} is not fitted. It can not transform anything".
                format(type(self).__name__))

        if self.mode_ == "covariants":
            new_even = np.empty([
                first_even.covariants_.shape[0],
                self.new_even_size_,
                self.l_max_ + 1,
                2 * self.l_max_ + 1,
            ])
            new_odd = np.empty([
                first_even.covariants_.shape[0],
                self.new_odd_size_,
                self.l_max_ + 1,
                2 * self.l_max_ + 1,
            ])
        else:
            new_even = np.empty(
                [first_even.covariants_.shape[0], self.new_even_size_, 1])
            new_odd = np.empty(
                [first_even.covariants_.shape[0], self.new_odd_size_, 1])

        if self.mode_ == "covariants":
            new_even_actual_sizes = np.zeros([self.l_max_ + 1], dtype=np.int32)
            new_odd_actual_sizes = np.zeros([self.l_max_ + 1], dtype=np.int32)
        else:
            new_even_actual_sizes = np.zeros([1], dtype=np.int32)
            new_odd_actual_sizes = np.zeros([1], dtype=np.int32)

        do_partial_expansion(
            self.clebsch_.precomputed_,
            first_even.covariants_,
            second_even.covariants_,
            self.l_max_,
            self.task_even_even_[0],
            new_even,
            new_even_actual_sizes,
            self.mode_,
            num_threads=self.num_threads_,
        )
        # print(new_even_actual_sizes)
        do_partial_expansion(
            self.clebsch_.precomputed_,
            first_odd.covariants_,
            second_odd.covariants_,
            self.l_max_,
            self.task_odd_odd_[0],
            new_even,
            new_even_actual_sizes,
            self.mode_,
            num_threads=self.num_threads_,
        )
        # print(new_even_actual_sizes)
        do_partial_expansion(
            self.clebsch_.precomputed_,
            first_even.covariants_,
            second_odd.covariants_,
            self.l_max_,
            self.task_even_odd_[0],
            new_odd,
            new_odd_actual_sizes,
            self.mode_,
            num_threads=self.num_threads_,
        )

        do_partial_expansion(
            self.clebsch_.precomputed_,
            first_odd.covariants_,
            second_even.covariants_,
            self.l_max_,
            self.task_odd_even_[0],
            new_odd,
            new_odd_actual_sizes,
            self.mode_,
            num_threads=self.num_threads_,
        )
        if self.mode_ == "covariants":
            for lambd in range(self.l_max_ + 1):
                if (self.even_multipliers_ is not None):
                    #print(new_even_actual_sizes)
                    #print(self.even_multipliers_[lambd].shape)
                    new_even[:, :new_even_actual_sizes[lambd], lambd, :(2 * lambd + 1)] = new_even[:, :new_even_actual_sizes[lambd], lambd, :(2 * lambd + 1)] * \
                                                                                          (self.even_multipliers_[lambd])[np.newaxis, :, np.newaxis]
                if (self.odd_multipliers_ is not None):
                    new_odd[:, :new_odd_actual_sizes[lambd], lambd, :(2 * lambd + 1)] = new_odd[:, :new_odd_actual_sizes[lambd], lambd,:(2 * lambd + 1)] * \
                                                                                        (self.odd_multipliers_[lambd])[np.newaxis, :, np.newaxis]

            return Data(new_even,
                        new_even_actual_sizes), Data(new_odd,
                                                     new_odd_actual_sizes)
        else:
            if (self.even_multipliers_ is not None):
                new_even[:, :new_even_actual_sizes[0], 0] = new_even[:, :new_even_actual_sizes[0], 0] * self.even_multipliers_[np.newaxis, :]
            if (self.odd_multipliers_ is not None):
                new_odd[:, :new_odd_actual_sizes[0], 0] = new_odd[:, :new_odd_actual_sizes[0],
                                                            0] * self.odd_multipliers_[np.newaxis, :]
            return (
                new_even[:, :new_even_actual_sizes[0], 0],
                new_odd[:, :new_odd_actual_sizes[0], 0]
            )

    def is_fitted(self):
        return self.fitted_
