from mdt.components_config.compartment_models import CompartmentConfig
from mdt.components_loader import CompartmentModelsLoader, bind_function

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GDRCylinders(CompartmentConfig):

    parameter_list = ('g', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'gamma_k', 'gamma_beta', 'gamma_nmr_cyl')
    dependency_list = (CompartmentModelsLoader().load('CylinderGPD'),)

    @bind_function
    def get_extra_results_maps(self, results_dict):
        return self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                            results_dict[self.name + '.phi'])
