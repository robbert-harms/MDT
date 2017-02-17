from mdt.components_config.compartment_models import CompartmentConfig


class SSFP_Ball(CompartmentConfig):

    parameter_list = ('d', 'delta', 'G', 'TR', 'flip_angle', 'b1_static', 'T1', 'T2')
    dependency_list = ('SSFP',)
    cl_code = '''
        return SSFP(d, delta, G, TR, flip_angle, b1_static, T1, T2);
    '''