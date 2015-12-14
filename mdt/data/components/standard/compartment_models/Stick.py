from mdt.models.compartments import DMRICompartmentModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Stick(DMRICompartmentModelBuilder):

    config = dict(
        name='Stick',
        cl_function_name='cmStick',
        parameter_list=('g', 'b', 'd', 'theta', 'phi'),
        cl_code_inline='''
            return exp(-b * d * pown(dot(g, (MOT_FLOAT_TYPE4)(cos(phi) * sin(theta),
                                                              sin(phi) * sin(theta), cos(theta), 0.0)), 2));
        '''
    )

    def get_extra_results_maps(self, results_dict):
        return self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])
