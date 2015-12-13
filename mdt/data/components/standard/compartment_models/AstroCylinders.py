from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.models.compartment_models import DMRICompartmentModelFunction
from mdt.components_loader import LibraryFunctionsLoader

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AstroCylinders(DMRICompartmentModelFunction):

    def __init__(self, name='AstroCylinders'):
        lib_loader = LibraryFunctionsLoader()

        super(AstroCylinders, self).__init__(
            name,
            'cmAstroCylinders',
            (get_parameter('g'),
             get_parameter('b'),
             get_parameter('G'),
             get_parameter('Delta'),
             get_parameter('delta'),
             get_parameter('d'),
             get_parameter('R')),
            resource_filename(__name__, 'AstroCylinders.h'),
            resource_filename(__name__, 'AstroCylinders.cl'),
            (lib_loader.load('MRIConstants'),
             lib_loader.load('NeumannCylPerpPGSESum'))
        )
