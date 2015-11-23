from pkg_resources import resource_filename
from mot.base import LibraryFunction, LibraryParameter, CLDataType

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NeumannCylPerpPGSESum(LibraryFunction):

    def __init__(self):
        super(NeumannCylPerpPGSESum, self).__init__(
            'double',
            'NeumannCylPerpPGSE',
            (LibraryParameter(CLDataType.from_string('model_float'), 'Delta'),
             LibraryParameter(CLDataType.from_string('model_float'), 'delta'),
             LibraryParameter(CLDataType.from_string('model_float'), 'd'),
             LibraryParameter(CLDataType.from_string('model_float'), 'R'),
             LibraryParameter(CLDataType.from_string('model_float*'), 'CLJnpZeros'),
             LibraryParameter(CLDataType.from_string('int'), 'CLJnpZerosLength')),
            resource_filename(__name__, 'NeumannCylPerpPGSESum.h'),
            resource_filename(__name__, 'NeumannCylPerpPGSESum.cl'),
            {},
            ())