from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco.Lagos'

"""MPM fitting (Weiskopf, 2016 ESMRMB Workshop)

This fitting is a model published by Helms (2008) and Weiskopf (2011) to determinate biological properties
of the tissue/sample in function *of several images*, which includes T1w, PDw and MTw images. This function is still an
approximation and, if the assumptions of those approximations hold for ex-vivo tissue, then can be used
in this data.
"""


class MPM_Fit(CompartmentConfig):

    parameter_list = ('TR', 'flip_angle', 'excitation_b1_map', 'T1')
    cl_code = 'return (flip_angle * excitation_b1_map) * ( (TR / T1) / ( pown(flip_angle * excitation_b1_map, 2) / 2 + ( TR / T1 ) ) );'
