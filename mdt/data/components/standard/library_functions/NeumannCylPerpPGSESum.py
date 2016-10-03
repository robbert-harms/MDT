from pkg_resources import resource_filename
from mot.cl_data_type import CLDataType
from mot.model_building.cl_functions.base import LibraryFunction
from mot.model_building.cl_functions.parameters import LibraryParameter

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NeumannCylPerpPGSESum(LibraryFunction):

    def __init__(self):
        super(NeumannCylPerpPGSESum, self).__init__(
            'double',
            'NeumannCylPerpPGSE',
            (LibraryParameter(CLDataType.from_string('mot_float_type'), 'Delta'),
             LibraryParameter(CLDataType.from_string('mot_float_type'), 'delta'),
             LibraryParameter(CLDataType.from_string('mot_float_type'), 'd'),
             LibraryParameter(CLDataType.from_string('mot_float_type'), 'R')),
            resource_filename(__name__, 'NeumannCylPerpPGSESum.h'),
            resource_filename(__name__, 'NeumannCylPerpPGSESum.cl'),
            {},
            ())
