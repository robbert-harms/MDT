from mdt.models.compartments import CompartmentConfig
from mdt.components_loader import bind_function


class SSFP_Stick(CompartmentConfig):

    parameter_list = ('g', 'd', 'TR', 'flip_angle', 'b1map', 'T1map', 'T2map')
    dependency_list = ('MRIConstants',)

    @bind_function
    def get_extra_results_maps(self, results_dict):
        return self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                            results_dict[self.name + '.phi'])
