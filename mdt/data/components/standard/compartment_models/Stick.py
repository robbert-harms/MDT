from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.models.compartment_models import DMRICompartmentModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Stick(DMRICompartmentModelBuilder):

    config = dict(
        name='Stick',
        cl_function_name='cmStick',
        parameter_list=('g', 'b', 'd', 'theta', 'phi'),
        cl_header_file=resource_filename(__name__, 'Stick.h'),
        cl_code_file=resource_filename(__name__, 'Stick.cl'),
    )

    def get_extra_results_maps(self, results_dict):
        return self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])

