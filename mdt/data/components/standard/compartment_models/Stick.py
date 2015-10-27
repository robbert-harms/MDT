from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.utils import DMRICompartmentModelFunction

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Stick(DMRICompartmentModelFunction):

    def __init__(self, name='Stick'):
        super(Stick, self).__init__(
            name,
            'cmStick',
            (get_parameter('g'),
             get_parameter('b'),
             get_parameter('d'),
             get_parameter('theta'),
             get_parameter('phi')),
            resource_filename(__name__, 'Stick.h'),
            resource_filename(__name__, 'Stick.cl'),
            ()
        )

    def get_extra_results_maps(self, results_dict):
        return self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])

