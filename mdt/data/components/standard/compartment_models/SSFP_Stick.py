from mdt.components_loader import bind_function
from mdt.models.compartments import CompartmentConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SSFP_Stick(CompartmentConfig):

    name = 'SSFP_Stick'
    cl_function_name = 'cmSSFP_Stick'
    parameter_list = ('g', 'd', 'TR', 'flip_angle', 'b1map', 'T1map', 'T2map')
    cl_code = CLCodeFromInlineString('''
        return exp(-b * d * pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta),
                                                          sin(phi) * sin(theta), cos(theta), 0.0)), 2));
    ''')

    @bind_function
    def get_extra_results_maps(self, results_dict):
        return self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                            results_dict[self.name + '.phi'])
