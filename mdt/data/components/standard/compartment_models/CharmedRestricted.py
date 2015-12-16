from mdt.components_loader import bind_function
from mdt.models.compartments import CompartmentConfig, CLCodeFromAdjacentFile

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CharmedRestricted(CompartmentConfig):

    name = 'CharmedRestricted'
    cl_function_name = 'cmCharmedRestricted'
    parameter_list = ('g', 'b', 'GAMMA2_G2_delta2', 'TE', 'd', 'theta', 'phi')
    cl_code = CLCodeFromAdjacentFile(__name__)

    @bind_function
    def get_extra_results_maps(self, results_dict):
        return self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])
