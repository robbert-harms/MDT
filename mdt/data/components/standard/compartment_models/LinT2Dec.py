from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Francisco.Lagos'


class LinT2Dec(CompartmentTemplate):

    parameter_list = ('TE', 'R2')
    cl_code = 'return -TE * R2;'

