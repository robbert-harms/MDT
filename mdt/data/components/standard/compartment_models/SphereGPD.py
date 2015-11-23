from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.utils import DMRICompartmentModelFunction
from mdt.components_loader import LibraryFunctionsLoader

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SphereGPD(DMRICompartmentModelFunction):

    def __init__(self, name='SphereGPD'):
        lib_loader = LibraryFunctionsLoader()

        super(SphereGPD, self).__init__(
            name,
            'cmSphereGPD',
            (get_parameter('Delta'),
             get_parameter('delta'),
             get_parameter('d'),
             get_parameter('R')),
            resource_filename(__name__, 'SphereGPD.h'),
            resource_filename(__name__, 'SphereGPD.cl'),
            (lib_loader.load('MRIConstants'),)
        )