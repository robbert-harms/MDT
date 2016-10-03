from pkg_resources import resource_filename
from mot.model_building.cl_functions.base import LibraryFunction

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MRIConstants(LibraryFunction):

    def __init__(self):
        super(MRIConstants, self).__init__(
            '',
            '',
            (),
            resource_filename(__name__, 'MRIConstants.h'),
            resource_filename(__name__, 'MRIConstants.cl'),
            {},
            ())
