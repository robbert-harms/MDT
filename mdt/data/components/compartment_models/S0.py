from mot.cl_functions import Scalar
from mot.parameter_functions.proposals import GaussianProposal
from mot.parameter_functions.transformations import ClampTransform
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class S0(Scalar):

    def __init__(self, name='S0', value=1e4, lower_bound=0, upper_bound=1e8):
        super(S0, self).__init__(name=name, value=value, lower_bound=lower_bound, upper_bound=upper_bound)
        self.parameter_list[0].name = 's0'
        self.parameter_list[0].parameter_transform = ClampTransform()
        self.parameter_list[0].sampling_proposal = GaussianProposal(std=25.0)
        self.parameter_list[0].perturbation_function = lambda v: np.clip(
            v + np.random.normal(scale=1e3, size=v.shape), lower_bound, upper_bound)