from mdt.components_config.library_functions import LibraryFunctionConfig


__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SSFP(LibraryFunctionConfig):

    description = '''
        Following Buxton equation and Miller paper (2008), the equation receives the following variables:

        Args:
            g: gradient vector (unity)
            d: diffusivity of the stick (eigenvalue, m^2/s)
            delta: gradient diffusion duration (s)
            G: gradient amplitude T/m
            TR: repetition time
            flip_angle: the excitation angle
            b1: taken from a b1+ map
            T1: longitudinal relaxation time
            T2: transversal relaxation time
    '''
    return_type = 'mot_float_type'
    parameter_list = ['d', 'delta', 'G', 'TR', 'flip_angle', 'b1', 'T1', 'T2']
    dependency_list = ('MRIConstants',)