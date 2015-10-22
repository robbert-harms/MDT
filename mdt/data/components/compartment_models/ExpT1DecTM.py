from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.utils import DMRICompartmentModelFunction

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT1DecTM(DMRICompartmentModelFunction):

    def __init__(self, name='T1_TM'):
        super(ExpT1DecTM, self).__init__(
            name,
            'cmExpT1DecTM',
            (get_parameter('TM'), get_parameter('T1')),
            resource_filename(__name__, 'ExpT1DecTM.h'),
            resource_filename(__name__, 'ExpT1DecTM.cl'),
            ()
        )