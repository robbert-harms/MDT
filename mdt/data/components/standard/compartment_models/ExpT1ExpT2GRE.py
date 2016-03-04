from mdt.models.compartments import CompartmentConfig, CLCodeFromInlineString

__author__ = 'Francisco.Lagos'


class ExpT1ExpT2GRE(CompartmentConfig):

    name = 'ExpT1ExpT2GRE'
    cl_function_name = 'cmExpT1ExpT2GRE'
    parameter_list = ('TR', 'TE', 'flip_angle', 'T1', 'T2')
    cl_code = CLCodeFromInlineString("""
        return sin(flip_angle) * (1 - exp(-TR / T1)) / (1 - cos(flip_angle) * exp(-TR / T1)) * exp(-TE / T2);
    """)