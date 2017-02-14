from mdt.components_loader import bind_function
from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CHARMEDRestricted(CompartmentConfig):

    parameter_list = ('g', 'b', 'G', 'Delta', 'delta', 'TE', 'd', 'theta', 'phi')
    dependency_list = ('MRIConstants',)

    @bind_function
    def get_extra_results_maps(self, results_dict):
        return self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                            results_dict[self.name + '.phi'])
