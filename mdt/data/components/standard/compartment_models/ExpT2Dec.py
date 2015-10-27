from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.utils import DMRICompartmentModelFunction

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT2Dec(DMRICompartmentModelFunction):

    def __init__(self, name='T2'):
        super(ExpT2Dec, self).__init__(
            name,
            'cmExpT2Dec',
            (get_parameter('TE'),
             get_parameter('T2')),
            resource_filename(__name__, 'ExpT2Dec.h'),
            resource_filename(__name__, 'ExpT2Dec.cl'),
            ()
        )