from mdt.components_config.compartment_models import CompartmentConfig


class SSFP_Ball(CompartmentConfig):

    parameter_list = ('d', 'delta', 'G', 'TR', 'flip_angle', 'b1_static', 'T1_static', 'T2_static')
    dependency_list = ('SSFP',)
    cl_code = '''
        return SSFP(d, delta, G, TR, flip_angle, b1_static, T1_static, T2_static);
    '''
