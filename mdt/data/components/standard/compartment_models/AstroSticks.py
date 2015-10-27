from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.utils import DMRICompartmentModelFunction

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AstroSticks(DMRICompartmentModelFunction):

    def __init__(self, name='AstroSticks'):
        super(AstroSticks, self).__init__(
            name,
            'cmAstroSticks',
            (get_parameter('g'),
             get_parameter('G'),
             get_parameter('b'),
             get_parameter('d')),
            resource_filename(__name__, 'AstroSticks.h'),
            resource_filename(__name__, 'AstroSticks.cl'),
            ()
        )