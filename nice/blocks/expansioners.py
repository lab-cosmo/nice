from nice.thresholding import get_thresholded_tasks
from nice.nice_utilities import do_partial_expansion, Data, get_sizes
from nice.clebsch_gordan import ClebschGordan, check_clebsch_gordan
import numpy as np
from sklearn.exceptions import NotFittedError


class ThresholdExpansioner:
    def __init__(self, num_expand=None, mode="covariants", num_threads=None):
        if num_expand is None:
            self.num_expand_ = -1
        else:
            self.num_expand_ = num_expand

        self.mode_ = mode
        self.num_threads_ = num_threads
        self.fitted_ = False

    def fit(self,
            first_even,
            first_odd,
            second_even,
            second_odd,
            clebsch_gordan=None):

        self.l_max_ = first_even.covariants_.shape[2] - 1
        
        if (first_even.importances_ is None) or (first_odd.importances_ is None) \
        or (second_even.importances_ is None) or (second_odd.importances_ is None):
            raise ValueError("For thresholding importances of features should be specified")
            
            
        (
            self.task_even_even_,
            self.task_odd_odd_,
            self.task_even_odd_,
            self.task_odd_even_,
        ) = get_thresholded_tasks(
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

        self.new_even_raw_importances_ = np.concatenate(
            [self.task_even_even_[1], self.task_odd_odd_[1]], axis=0)
        self.new_odd_raw_importances_ = np.concatenate(
            [self.task_even_odd_[1], self.task_odd_even_[1]], axis=0)
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
            return Data(new_even,
                        new_even_actual_sizes), Data(new_odd,
                                                     new_odd_actual_sizes)
        else:
            return (
                new_even[:, :new_even_actual_sizes[0], 0],
                new_odd[:, :new_odd_actual_sizes[0], 0],
            )

    def is_fitted(self):
        return self.fitted_
