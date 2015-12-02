from pkg_resources import resource_filename
from mot.base import LibraryFunction, LibraryParameter, DataType

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NeumannCylPerpPGSESum(LibraryFunction):

    def __init__(self):
        super(NeumannCylPerpPGSESum, self).__init__(
            'double',
            'NeumannCylPerpPGSE',
            (LibraryParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'Delta'),
             LibraryParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'delta'),
             LibraryParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'd'),
             LibraryParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'R')),
            resource_filename(__name__, 'NeumannCylPerpPGSESum.h'),
            resource_filename(__name__, 'NeumannCylPerpPGSESum.cl'),
            {},
            ())
