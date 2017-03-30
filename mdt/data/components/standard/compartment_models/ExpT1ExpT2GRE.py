from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco.Lagos'


class ExpT1ExpT2GRE(CompartmentConfig):

    parameter_list = ('TR', 'TE', 'flip_angle', 'T1', 'T2')
    cl_code = """
        return sin(flip_angle) * (1 - exp(-TR / T1)) / (1 - cos(flip_angle) * exp(-TR / T1)) * exp(-TE / T2);
    """
