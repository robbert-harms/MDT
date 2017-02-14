from mdt.components_config.compartment_models import CompartmentConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AstroCylinders(CompartmentConfig):

    parameter_list = ('g', 'b', 'G', 'Delta', 'delta', 'd', 'R')
    dependency_list = ['MRIConstants',
                       'NeumannCylPerpPGSESum']
    cl_code = '''
        mot_float_type sum = NeumannCylPerpPGSESum(Delta, delta, d, R);

        mot_float_type lperp = (-2 * GAMMA_H_SQ * sum);
        mot_float_type lpar = -b * 1.0/pown(G, 2) * d;

        return (sqrt(M_PI) / (2 * G * sqrt(lperp - lpar)))
                    * exp(pown(G, 2) * lperp)
                    * erf(G * sqrt(lperp - lpar));
    '''
