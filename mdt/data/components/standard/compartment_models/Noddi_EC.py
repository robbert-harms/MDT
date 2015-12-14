from mdt.models.compartments import DMRICompartmentModelBuilder
from mot.cl_functions import CerfDawson

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Noddi_EC(DMRICompartmentModelBuilder):

    config = dict(
        name='Noddi_EC',
        cl_function_name='cmNoddi_EC',
        parameter_list=('g', 'b', 'd', 'dperp0', 'theta', 'phi', 'kappa'),
        module_name=__name__,
        dependency_list=(CerfDawson(),)
    )

    def get_extra_results_maps(self, results_dict):
        return self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])
