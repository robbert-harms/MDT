from mdt.models.parameters import StaticMapParameterConfig

__author__ = 'Robbert Harms'
__date__ = "2016-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


"""Static data parameters.

These parameters are in usage similar to fixed free parameters. They are defined as static data parameters to
make clear that they are meant to carry additional observational data about a problem.

Please choose the parameter type for a model and parameter carefully since the type signifies how the parameter and
its data are handled during model construction.

"""


class b1_static(StaticMapParameterConfig):

    name = 'b1_static'
    data_type = 'mot_float_type'
    value = 1

class fa_static(StaticMapParameterConfig):

    name = 'fa_static'
    data_type = 'mot_float_type'
    value = 1

"""" S_weighted. This parameter is created *only for linear T1 decay fitting* of GRE data with variable flip angle. S_weighted
is defined as the input data divided by the tan(flip_angle) ->  S_weighted = data / tan (flip_angle * B1_map)
"""""

class Sw_static(StaticMapParameterConfig):

    name = 'Sw_static'
    data_type = 'mot_float_type'
    value = 1


class T1_static(StaticMapParameterConfig):

    name = 'T1_static'
    data_type = 'mot_float_type'


class T2_static(StaticMapParameterConfig):

    name = 'T2_static'
    data_type = 'mot_float_type'


class T2s_static(StaticMapParameterConfig):

    name = 'T2s_static'
    data_type = 'mot_float_type'

#class TM_0(StaticMapParameterConfig):

#    name = 'TM_0'
#    data_type = 'mot_float_type'

