from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco.Lagos'

"""Lineal T1 fitting (Weiskopf, 2016 ESMRMB Workshop)

This fitting is the extension of the standard GRE equation for flip angles lower than 90deg. This modelling allows a
linear fitting of the data if is enough data to support it. In principle, it should not be a problem if only two
points are used, however the addition of a constant in the equation could give some kind of uncertainty.

B1 has to be normalized *in function of the reference voltage, the angle distribution and the reference angle*.
Here I assume that TR <<< T1, then exp(-TR/T1) ~ 1 - TR/T1. Then the equation becomes 'simpler'. However, fi this condition
is not achieved, then return to the standard equation.
Also, DATA HAS TO BE PROCESSED BEFORE TO USE THIS EQUATION. Please apply log() on the data.
"""


class LinT1GRE(CompartmentConfig):

    parameter_list = ('Sw_static', 'E1')
    #cl_code = """
    #    return sin( B1 * angle ) / ( 1 - cos( B1 * angle ) * exp( - TR / T1 ));
    #"""
    cl_code = """
        return Sw_static * E1;
    """
