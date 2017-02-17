from mdt.components_config.compartment_models import CompartmentConfig
from mdt.components_loader import bind_function


class SSFP_Stick(CompartmentConfig):

    parameter_list = ('g', 'd', 'theta', 'phi', 'delta', 'G', 'TR', 'flip_angle', 'b1_static', 'T1_exvivo', 'T2_exvivo')
    dependency_list = ('SSFP',)
    cl_code = '''
        const mot_float_type direction_2 = pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta), sin(phi) * sin(theta),
                                                                         cos(theta), 0.0)), 2);

        return SSFP(d * direction_2, delta, G, TR, flip_angle, b1_static, T1_exvivo, T2_exvivo);
    '''

    @bind_function
    def get_extra_results_maps(self, results_dict):
        return self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                            results_dict[self.name + '.phi'])