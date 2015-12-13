from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.models.compartment_models import DMRICompartmentModelFunction
from mot.cl_functions import CerfDawson

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Noddi_EC(DMRICompartmentModelFunction):

    def __init__(self, name='Noddi_EC'):
        super(Noddi_EC, self).__init__(
            name,
            'cmNoddi_EC',
            (get_parameter('g'),
             get_parameter('b'),
             get_parameter('d'),
             get_parameter('dperp0'),
             get_parameter('theta'),
             get_parameter('phi'),
             get_parameter('kappa')),
            resource_filename(__name__, 'Noddi_EC.h'),
            resource_filename(__name__, 'Noddi_EC.cl'),
            (CerfDawson(),)
        )

    def get_extra_results_maps(self, results_dict):
        return self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])
