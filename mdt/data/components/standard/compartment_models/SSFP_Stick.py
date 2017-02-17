from mdt.components_config.compartment_models import CompartmentConfig
from mdt.components_loader import bind_function


class SSFP_Stick(CompartmentConfig):

    parameter_list = ('g', 'd', 'theta', 'phi', 'delta', 'G', 'TR', 'flip_angle', 'b1_static', 'T1', 'T2')
    dependency_list = ('SSFP',)
    cl_code = '''
        mot_float_type adc = d * pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta), sin(phi) * sin(theta),
                                                               cos(theta), 0.0)), 2);

        return SSFP(adc, delta, G, TR, flip_angle, b1_static, T1, T2);
    '''

    @bind_function
    def get_extra_results_maps(self, results_dict):
        return self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                            results_dict[self.name + '.phi'])