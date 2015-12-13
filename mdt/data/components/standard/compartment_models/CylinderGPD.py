from mdt.models.compartment_models import DMRICompartmentModelBuilder
from mdt.components_loader import LibraryFunctionsLoader

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lib_loader = LibraryFunctionsLoader()


class CylinderGPD(DMRICompartmentModelBuilder):

    config = dict(
        name='CylinderGPD',
        cl_function_name='cmCylinderGPD',
        parameter_list=('g', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'R'),
        module_name=__name__,
        dependency_list=[lib_loader.load('MRIConstants'),
                         lib_loader.load('NeumannCylPerpPGSESum')]
    )

    def get_extra_results_maps(self, results_dict):
        return self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])
