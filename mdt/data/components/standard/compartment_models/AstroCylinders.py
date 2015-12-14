from mdt.models.compartments import DMRICompartmentModelBuilder
from mdt.components_loader import LibraryFunctionsLoader

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lib_loader = LibraryFunctionsLoader()


class AstroCylinders(DMRICompartmentModelBuilder):

    config = dict(
        name='AstroCylinders',
        cl_function_name='cmAstroCylinders',
        parameter_list=('g', 'b', 'G', 'Delta', 'delta', 'd', 'R'),
        module_name=__name__,
        dependency_list=[lib_loader.load('MRIConstants'),
                         lib_loader.load('NeumannCylPerpPGSESum')]
    )
