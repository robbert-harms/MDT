from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.utils import DMRICompartmentModelFunction
from mdt.components_loader import CompartmentModelsLoader

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GDRCylinders(DMRICompartmentModelFunction):

    def __init__(self, name='GDRCylinders'):
        compartment_loader = CompartmentModelsLoader()

        super(GDRCylinders, self).__init__(
            name,
            'cmGDRCylinders',
            (get_parameter('g'),
             get_parameter('G'),
             get_parameter('Delta'),
             get_parameter('delta'),
             get_parameter('d'),
             get_parameter('theta'),
             get_parameter('phi'),
             get_parameter('gamma_k'),
             get_parameter('gamma_beta'),
             get_parameter('gamma_nmr_cyl'),
             get_parameter('CLJnpZeros'),
             get_parameter('CLJnpZerosLength')),
            resource_filename(__name__, 'GDRCylinders.h'),
            resource_filename(__name__, 'GDRCylinders.cl'),
            (compartment_loader.load('CylinderGPD'),)
        )

    def get_extra_results_maps(self, results_dict):
        return self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])