from mdt.models.compartments import DMRICompartmentModelBuilder, CLCodeFromAdjacentFile
from mdt.components_loader import LibraryFunctionsLoader
from mot.cl_functions import FirstLegendreTerm, CerfErfi, CerfDawson
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lib_loader = LibraryFunctionsLoader()


class Noddi_IC(DMRICompartmentModelBuilder):

    config = dict(
        name='Noddi_IC',
        cl_function_name='cmNoddi_IC',
        parameter_list=('g', 'b', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'kappa', 'R'),
        cl_code=CLCodeFromAdjacentFile(__name__),
        dependency_list=(CerfDawson(), CerfErfi(), FirstLegendreTerm(),
                         lib_loader.load('MRIConstants'),
                         lib_loader.load('NeumannCylPerpPGSESum'))
    )

    def get_extra_results_maps(self, results_dict):
        maps = self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])
        maps.update({self.name + '.odi': np.arctan2(1.0, results_dict[self.name + '.kappa'] * 10) * 2 / np.pi})
        return maps
