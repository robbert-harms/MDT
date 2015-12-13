from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.models.compartment_models import DMRICompartmentModelFunction

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT1DecTR(DMRICompartmentModelFunction):

    def __init__(self, name='T1_IR'):
        super(ExpT1DecTR, self).__init__(
            name,
            'cmExpT1DecTR',
            (get_parameter('TR'), get_parameter('T1')),
            resource_filename(__name__, 'ExpT1DecTR.h'),
            resource_filename(__name__, 'ExpT1DecTR.cl'),
            ()
        )
