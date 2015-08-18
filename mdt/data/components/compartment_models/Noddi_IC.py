from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.utils import DMRICompartmentModelFunction
from mdt.components_loader import LibraryFunctionsLoader
from mot.cl_functions import FirstLegendreTerm, CerfErfi, CerfDawson
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Noddi_IC(DMRICompartmentModelFunction):

    def __init__(self, name='Noddi_IC'):
        lib_loader = LibraryFunctionsLoader()

        super(Noddi_IC, self).__init__(
            name,
            'cmNoddi_IC',
            (get_parameter('g'),
             get_parameter('b'),
             get_parameter('G'),
             get_parameter('Delta'),
             get_parameter('delta'),
             get_parameter('d'),
             get_parameter('theta'),
             get_parameter('phi'),
             get_parameter('kappa'),
             get_parameter('R'),
             get_parameter('CLJnpZeros'),
             get_parameter('CLJnpZerosLength')
             ),
            resource_filename(__name__, 'Noddi_IC.h'),
            resource_filename(__name__, 'Noddi_IC.cl'),
            (CerfDawson(), CerfErfi(), FirstLegendreTerm(), 
             lib_loader.load('MRIConstants'),
             lib_loader.load('NeumannCylPerpPGSESum'))
        )

    def get_extra_results_maps(self, results_dict):
        maps = self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])
        maps.update({self.name + '.odi': np.arctan2(1.0, results_dict[self.name + '.kappa']) * 2/np.pi})
        return maps