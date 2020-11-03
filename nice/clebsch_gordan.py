from sympy.physics.wigner import clebsch_gordan
from sympy import S
import numpy as np


def get_single(l1, l2, l, m1, m2):
    return float(clebsch_gordan(S(l1), S(l2), S(l), S(m1), S(m2), S(m1 + m2)))


class ClebschGordan:
    def __init__(self, l_max):
        self.l_max_ = l_max
        self.precomputed_ = np.zeros(
            [l_max + 1, l_max + 1, l_max + 1, 2 * l_max + 1, 2 * l_max + 1])

        for l1 in range(l_max + 1):
            for l2 in range(l_max + 1):
                for l in range(l_max + 1):
                    for m1 in range(-l_max, l_max + 1):
                        for m2 in range(-l_max, l_max + 1):
                            now = get_single(l1, l2, l, m1, m2)
                            self.precomputed_[l1, l2, l, m1 + l1,
                                              m2 + l2] = now


def check_clebsch_gordan(clebsch_gordan, required_l_max):
    if (type(clebsch_gordan) != ClebschGordan):
        raise ValueError("type of precomputed clebsch gordan "
                         "coefficients should be ClebschGordan class.")
    if (clebsch_gordan.l_max_ < required_l_max):
        raise ValueError("given precomputed clebsch gordan coefficients "
                         "have smaller l_max than required one.")
