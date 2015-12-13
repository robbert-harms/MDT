from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.models.compartment_models import DMRICompartmentModelFunction

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Zeppelin(DMRICompartmentModelFunction):

    def __init__(self, name='Zeppelin'):
        super(Zeppelin, self).__init__(
            name,
            'cmZeppelin',
            (get_parameter('g'),
             get_parameter('b'),
             get_parameter('d'),
             get_parameter('dperp0'),
             get_parameter('theta'),
             get_parameter('phi')),
            resource_filename(__name__, 'Zeppelin.h'),
            resource_filename(__name__, 'Zeppelin.cl'),
            ()
        )
