from mdt.models.compartments import CompartmentConfig, CLCodeFromAdjacentFile, CLCodeFromInlineString
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
    cl_code = CLCodeFromInlineString('''
        MOT_FLOAT_TYPE sum = NeumannCylPerpPGSESum(Delta, delta, d, R);

        MOT_FLOAT_TYPE lperp = (-2 * GAMMA_H_SQ * sum);
        MOT_FLOAT_TYPE lpar = -b * 1.0/pown(G, 2) * d;

        return (sqrt(M_PI) / (2 * G * sqrt(lperp - lpar)))
                    * exp(pown(G, 2) * lperp)
                    * erf(G * sqrt(lperp - lpar));
    ''')
