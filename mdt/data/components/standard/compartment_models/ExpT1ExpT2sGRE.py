from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco.Lagos'


class ExpT1ExpT2sGRE(CompartmentConfig):

    parameter_list = ('TR', 'TE', 'flip_angle', 'T1', 'T2s')
    cl_code = """
        return sin(flip_angle) * (1 - exp(-TR / T1)) / (1 - cos(flip_angle) * exp(-TR / T1)) * exp(-TE / T2s);
    """
