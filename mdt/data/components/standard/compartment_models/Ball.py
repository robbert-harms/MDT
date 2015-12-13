from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.models.compartment_models import DMRICompartmentModelFunction

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Ball(DMRICompartmentModelFunction):

    def __init__(self, name='Ball'):
        super(Ball, self).__init__(
            name,
            'cmBall',
            (get_parameter('b'), get_parameter('d')),
            resource_filename(__name__, 'Ball.h'),
            resource_filename(__name__, 'Ball.cl'),
            ()
        )
