from mdt.models.compartments import CompartmentConfig, CLCodeFromAdjacentFile, CLCodeFromInlineString
from mdt.components_loader import LibraryFunctionsLoader, bind_function

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lib_loader = LibraryFunctionsLoader()


class CylinderGPD(CompartmentConfig):

    name = 'CylinderGPD'
    cl_function_name = 'cmCylinderGPD'
    parameter_list = ('g', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'R')
    dependency_list = [lib_loader.load('MRIConstants'),
                       lib_loader.load('NeumannCylPerpPGSESum')]
    cl_code = CLCodeFromInlineString('''
        MOT_FLOAT_TYPE sum = NeumannCylPerpPGSESum(Delta, delta, d, R);

        const MOT_FLOAT_TYPE4 n = (MOT_FLOAT_TYPE4)(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta), 0.0);
        MOT_FLOAT_TYPE omega = (G == 0.0) ? M_PI_2 : acos(dot(n, g * G) / (G * length(n)));

        return exp(-2 * GAMMA_H_SQ * pown(G * sin(omega), 2) * sum) *
                exp(-(Delta - (delta/3.0)) * pown(GAMMA_H * delta * G * cos(omega), 2) * d);
    ''')

    @bind_function
    def get_extra_results_maps(self, results_dict):
        return self._get_vector_result_maps(results_dict[self.name + '.theta'],
                                            results_dict[self.name + '.phi'])
