from mdt.components_config.library_functions import LibraryFunctionConfig


__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SSFP(LibraryFunctionConfig):

    description = '''
        Implementation of the SSFP signal attenuation.

        This uses the algorithm described in:
        "The Diffusion Sensitivity of Fast Steady-State Free Precession Imaging", Richard B. Buxton (1993)

        Args:
            d: diffusivity or Apparent Diffusion Coefficient
            delta: gradient diffusion duration (s)
            G: gradient amplitude T/m
            TR: repetition time
            flip_angle: the excitation angle
            b1: taken from a b1+ map
            T1: longitudinal relaxation time
            T2: transversal relaxation time
    '''
    return_type = 'double'
    parameter_list = ['d', 'delta', 'G', 'TR', 'flip_angle', 'b1', 'T1', 'T2']
    dependency_list = ('MRIConstants',)
