from mdt.models.compartments import DMRICompartmentModelBuilder, CLCodeFromAdjacentFile
from mdt.components_loader import CompartmentModelsLoader

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


compartment_loader = CompartmentModelsLoader()


class GDRCylinders(DMRICompartmentModelBuilder):

    config = dict(
        name='GDRCylinders',
        cl_function_name='cmGDRCylinders',
        parameter_list=('g', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'gamma_k', 'gamma_beta', 'gamma_nmr_cyl'),
        cl_code=CLCodeFromAdjacentFile(__name__),
        dependency_list=(compartment_loader.load('CylinderGPD'),)
    )

    def get_extra_results_maps(self, results_dict):
        return self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])
