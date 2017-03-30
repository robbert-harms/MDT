from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco.Lagos'


class LinT2Dec(CompartmentConfig):

    parameter_list = ('TE', 'R2')
    cl_code = 'return -TE * R2;'

