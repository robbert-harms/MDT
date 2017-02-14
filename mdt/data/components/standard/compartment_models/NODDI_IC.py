from mdt.components_config.compartment_models import CompartmentConfig
from mdt.components_loader import bind_function
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDI_IC(CompartmentConfig):

    parameter_list = ('g', 'b', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'kappa', 'R')
    dependency_list = ('CerfErfi',
                       'MRIConstants',
                       'NeumannCylPerpPGSESum')

    @bind_function
    def get_extra_results_maps(self, results_dict):
        maps = self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                            results_dict[self.name + '.phi'])
        maps.update({self.name + '.odi': np.arctan2(1.0, results_dict[self.name + '.kappa'] * 10) * 2 / np.pi})
        return maps
