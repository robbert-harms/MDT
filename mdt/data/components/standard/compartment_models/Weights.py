from mdt import WeightCompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = '2018-03-23'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class Weight(WeightCompartmentTemplate):

    parameters = ('w',)
    cl_code = 'return w;'


class ARD_Beta_Weight(Weight):
    """A weight compartment with a Beta distribution prior between [0, 1].

    Meant for use in Automatic Relevance Detection MCMC sampling.
    """
    parameters = ('w_ard_beta(w)',)


class ARD_Gaussian_Weight(Weight):
    """A weight compartment with a Gaussian prior with mean at zero and std given by a hyperparameter.

    Meant for use in Automatic Relevance Detection MCMC sampling.
    """
    parameters = ('w_ard_gaussian(w)',)
