from mdt.components_loader import LibraryFunctionsLoader
from mdt.models.compartments import DMRICompartmentModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lib_loader = LibraryFunctionsLoader()


class SphereGPD(DMRICompartmentModelBuilder):

    config = dict(
        name='SphereGPD',
        cl_function_name='cmSphereGPD',
        parameter_list=('Delta', 'delta', 'd', 'R'),
        module_name=__name__,
        dependency_list=(lib_loader.load('MRIConstants'),)
    )
