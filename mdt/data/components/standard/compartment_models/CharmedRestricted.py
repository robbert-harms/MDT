from mdt.components_loader import bind_function
from mdt.models.compartments import CompartmentConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CharmedRestricted(CompartmentConfig):

    parameter_list = ('g', 'b', 'q', 'Delta', 'delta', 'TE', 'd', 'theta', 'phi')

    @bind_function
    def get_extra_results_maps(self, results_dict):
        return self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                            results_dict[self.name + '.phi'])
