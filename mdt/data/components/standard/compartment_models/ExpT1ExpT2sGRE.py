from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Francisco.Lagos'


class ExpT1ExpT2sGRE(CompartmentTemplate):

    parameter_list = ('TR', 'TE', 'flip_angle', 'T1', 'T2s')
    cl_code = """
        return sin(flip_angle) * (1 - exp(-TR / T1)) / (1 - cos(flip_angle) * exp(-TR / T1)) * exp(-TE / T2s);
    """
