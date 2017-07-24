"""Definitions of the model data parameters.

These are meant for model specific data that the model needs to work. You can of course inline these variables in
the code for one of the models (which is faster), but this way lets the user change the specifics of the
model by changing the data in the model data parameters.

Please choose the parameter type for a model and parameter carefully since the type signifies how the parameter and
its data are handled during model construction.
"""
from mdt.component_templates.parameters import ModelDataParameterTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GDRCylinders_nmr_bins(ModelDataParameterTemplate):

    data_type = 'int'
    value = 5
