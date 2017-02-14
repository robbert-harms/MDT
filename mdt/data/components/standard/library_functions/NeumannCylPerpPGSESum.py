from mdt.components_config.library_functions import LibraryFunctionConfig


__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NeumannCylPerpPGSESum(LibraryFunctionConfig):

    description = '''
        This function returns the summation of the signal attenuation in perpendicular direction (LePerp)
        for Radius R, according to the Neumann model.

        The summation is the sum over the Bessel roots up to a accuracy of 1e-8, it does not
        calculate the complete signal for a cylinder dMRI compartment model.
    '''
    return_type = 'mot_float_type'
    parameter_list = ['Delta', 'delta', 'd', 'R']
