from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco.Lagos'

"""MPM fitting (Weiskopf, 2016 ESMRMB Workshop)

This fitting is a model published by Helms (2008) and Weiskopf (2011) to determinate biological properties
of the tissue/sample in function *of several images*, which includes T1w, PDw and MTw images. This function is still an
approximation and, if the assumptions of those approximations hold for ex-vivo tissue, then can be used
in this data.
"""


class LinMPM_Fit(CompartmentConfig):

    parameter_list = ('TR', 'flip_angle', 'b1_static', 'T1')
    cl_code = 'return log(flip_angle * b1_static) + log(TR / T1) - log( pown(flip_angle * b1_static, 2) / 2 + ( TR / T1 ) ) ;'
