from mdt.components_config.compartment_models import CompartmentConfig
from mdt.components_loader import bind_function


class SSFP_Zeppelin(CompartmentConfig):

    parameter_list = ('g', 'd', 'dperp0', 'theta', 'phi', 'delta', 'G', 'TR', 'flip_angle',
                      'b1_static', 'T1_exvivo', 'T2_exvivo')
    dependency_list = ('SSFP',)
    cl_code = '''
        mot_float_type adc = dperp0 + ((d - dperp0) * pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta),
                                                                   sin(phi) * sin(theta), cos(theta), 0.0)), 2);

        return SSFP(adc, delta, G, TR, flip_angle, b1_static, T1_exvivo, T2_exvivo);
    '''

    @bind_function
    def get_extra_results_maps(self, results_dict):
        return self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                            results_dict[self.name + '.phi'])