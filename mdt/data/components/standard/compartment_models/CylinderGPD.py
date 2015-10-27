from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.utils import DMRICompartmentModelFunction
from mdt.components_loader import LibraryFunctionsLoader

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CylinderGPD(DMRICompartmentModelFunction):

    def __init__(self, name='CylinderGPD'):
        lib_loader = LibraryFunctionsLoader()

        super(CylinderGPD, self).__init__(
            name,
            'cmCylinderGPD',
            (get_parameter('g'),
             get_parameter('G'),
             get_parameter('Delta'),
             get_parameter('delta'),
             get_parameter('d'),
             get_parameter('theta'),
             get_parameter('phi'),
             get_parameter('R'),
             get_parameter('CLJnpZeros'),
             get_parameter('CLJnpZerosLength')),
            resource_filename(__name__, 'CylinderGPD.h'),
            resource_filename(__name__, 'CylinderGPD.cl'),
            (lib_loader.load('MRIConstants'),
             lib_loader.load('NeumannCylPerpPGSESum'))
        )

    def get_extra_results_maps(self, results_dict):
        return self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])