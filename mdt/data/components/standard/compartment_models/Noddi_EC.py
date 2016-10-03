from mdt.components_loader import bind_function
from mdt.models.compartments import CompartmentConfig
from mot.model_building.cl_functions.library_functions import CerfDawson

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Noddi_EC(CompartmentConfig):

    parameter_list = ('g', 'b', 'd', 'dperp0', 'theta', 'phi', 'kappa')
    dependency_list = (CerfDawson(),)

    @bind_function
    def get_extra_results_maps(self, results_dict):
        return self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                            results_dict[self.name + '.phi'])
