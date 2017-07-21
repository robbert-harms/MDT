from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Francisco.Lagos'


class ExpT1DecGRE(CompartmentTemplate):

    parameter_list = ('TR', 'flip_angle', 'b1_static', 'T1')
    cl_code = """
        return sin(flip_angle * b1_static) * (1 - exp(-TR / T1)) / (1 - cos(flip_angle * b1_static) * exp(-TR / T1) );
    """
