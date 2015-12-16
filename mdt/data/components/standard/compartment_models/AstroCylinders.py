from mdt.models.compartments import CompartmentConfig, CLCodeFromAdjacentFile
from mdt.components_loader import LibraryFunctionsLoader

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lib_loader = LibraryFunctionsLoader()


class AstroCylinders(CompartmentConfig):

    name = 'AstroCylinders'
    cl_function_name = 'cmAstroCylinders'
    parameter_list = ('g', 'b', 'G', 'Delta', 'delta', 'd', 'R')
    dependency_list = [lib_loader.load('MRIConstants'),
                       lib_loader.load('NeumannCylPerpPGSESum')]
