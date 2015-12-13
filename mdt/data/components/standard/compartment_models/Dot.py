from pkg_resources import resource_filename
from mdt.models.compartment_models import DMRICompartmentModelFunction

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Dot(DMRICompartmentModelFunction):

    def __init__(self, name='Dot'):
        super(Dot, self).__init__(
            name,
            'cmDot',
            (),
            resource_filename(__name__, 'Dot.h'),
            resource_filename(__name__, 'Dot.cl'),
            ()
        )
