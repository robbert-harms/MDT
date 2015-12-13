from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.models.compartment_models import DMRICompartmentModelFunction
from mdt.components_loader import LibraryFunctionsLoader

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CharmedRestricted(DMRICompartmentModelFunction):

    def __init__(self, name='CharmedRestricted'):
        super(CharmedRestricted, self).__init__(
            name,
            'cmCharmedRestricted',
            (get_parameter('g'),
             get_parameter('b'),
             get_parameter('GAMMA2_G2_delta2'),
             get_parameter('TE'),
             get_parameter('d'),
             get_parameter('theta'),
             get_parameter('phi')),
            resource_filename(__name__, 'CharmedRestricted.h'),
            resource_filename(__name__, 'CharmedRestricted.cl'),
            ()
        )

    def get_extra_results_maps(self, results_dict):
        return self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])
