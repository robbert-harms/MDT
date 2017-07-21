from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Francisco.Lagos'


class ExpT1DecGRE(CompartmentConfig):

    parameter_list = ('TR', 'flip_angle', 'excitation_b1_map', 'T1')
    cl_code = """
        return sin((double)flip_angle * excitation_b1_map) *
              (1 - exp(-TR / (double)T1)) /
              (1 - cos((double)flip_angle * excitation_b1_map) * exp(-TR / (double)T1) );
    """
