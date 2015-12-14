from mdt.models.compartments import DMRICompartmentModelFunction

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CharmedRestricted(DMRICompartmentModelFunction):

    config = dict(
        name='CharmedRestricted',
        cl_function_name='cmCharmedRestricted',
        parameter_list=('g', 'b', 'GAMMA2_G2_delta2', 'TE', 'd', 'theta', 'phi'),
        module_name=__name__
    )

    def get_extra_results_maps(self, results_dict):
        return self._get_single_dir_coordinate_maps(results_dict[self.name + '.theta'],
                                                    results_dict[self.name + '.phi'],
                                                    results_dict[self.name + '.d'])
