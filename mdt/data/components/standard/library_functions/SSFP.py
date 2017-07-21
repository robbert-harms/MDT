from mdt.component_templates.library_functions import LibraryFunctionTemplate


__author__ = 'Robbert Harms, Francisco J. Fritz'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SSFP(LibraryFunctionTemplate):

    description = '''
        Implementation of the SSFP signal attenuation.

        This uses the algorithm described in:
        "The Diffusion Sensitivity of Fast Steady-State Free Precession Imaging", Richard B. Buxton (1993)

        Args:
            d: diffusivity or Apparent Diffusion Coefficient (m^2/s)
            delta: gradient diffusion duration (s)
            G: gradient amplitude (T/m)
            TR: repetition time (seconds)
            flip_angle: the excitation angle (radians)
            b1: taken from a b1+ map (a.u.)
            T1: longitudinal relaxation time (s)
            T2: transversal relaxation time (s)
    '''
    return_type = 'double'
    parameter_list = ['d', 'delta', 'G', 'TR', 'flip_angle', 'b1', 'T1', 'T2']
    dependency_list = ('MRIConstants',)
