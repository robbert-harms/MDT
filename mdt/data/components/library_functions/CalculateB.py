from pkg_resources import resource_filename
from mdt.components_loader import LibraryFunctionsLoader
from mot.base import LibraryFunction, LibraryParameter, CLDataType

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CalculateB(LibraryFunction):

    def __init__(self):
        lib_loader = LibraryFunctionsLoader()

        super(CalculateB, self).__init__(
            'double',
            'calculate_b',
            (LibraryParameter(CLDataType.from_string('double'), 'G'),
             LibraryParameter(CLDataType.from_string('double'), 'Delta'),
             LibraryParameter(CLDataType.from_string('double'), 'delta')),
            resource_filename(__name__, 'calculate_b.h'),
            resource_filename(__name__, 'calculate_b.cl'),
            {},
            (lib_loader.load('MRIConstants'), ))
