from mdt.models.compartments import CompartmentConfig
from mdt.components_loader import LibraryFunctionsLoader, bind_function
from mot.model_building.cl_functions.library_functions import CerfErfi
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lib_loader = LibraryFunctionsLoader()


class Noddi_IC(CompartmentConfig):

    parameter_list = ('g', 'b', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'kappa', 'R')
    dependency_list = (CerfErfi(),
                       lib_loader.load('MRIConstants'),
                       lib_loader.load('NeumannCylPerpPGSESum'))

    @bind_function
    def get_extra_results_maps(self, results_dict):
        maps = self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                            results_dict[self.name + '.phi'])
        maps.update({self.name + '.odi': np.arctan2(1.0, results_dict[self.name + '.kappa'] * 10) * 2 / np.pi})
        return maps
