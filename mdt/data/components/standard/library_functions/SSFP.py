from mdt import LibraryFunctionTemplate


__author__ = 'Robbert Harms, Francisco J. Fritz'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SSFP(LibraryFunctionTemplate):
    """Implementation of the SSFP signal attenuation.

    This uses the algorithm described in:
    "The Diffusion Sensitivity of Fast Steady-State Free Precession Imaging", Richard B. Buxton (1993)

    Args:
        d: diffusivity or Apparent Diffusion Coefficient (m^2/s)
        delta: gradient diffusion duration (s)
        G: gradient amplitude (T/m)
        TR: repetition time (seconds)
        flip_angle: the excitation angle (radians)
        b1: taken from a b1+ map (a.u.)
        T1: longitudinal relaxation time (s)
        T2: transversal relaxation time (s)
    """
    return_type = 'double'
    parameters = ['double d', 'double delta', 'double G', 'double TR',
                  'double flip_angle', 'double b1', 'double T1', 'double T2']
    dependencies = ('MRIConstants',)
    cl_code = '''
        double cos_b1_corrected_flip_angle;
        const double sin_b1_corrected_flip_angle = sincos(flip_angle * b1, &cos_b1_corrected_flip_angle);
    
        const double E1 = exp(-TR / T1);
        const double E2 = exp(-TR / T2);
    
        const double q_magnitude_2 = GAMMA_H_SQ * (double)(G * G) * (delta * delta);
    
        const double b = q_magnitude_2 * TR;
        const double beta = q_magnitude_2 * delta;
    
        const double A1 = exp(-b * d);
        const double A2 = exp(-beta * d);
    
        const double s = E2 * A1 * pow(A2, (double)(-4/3.0))
                                    * (1 - (E1 * cos_b1_corrected_flip_angle))
                                    + E2 * pow(A2, (double)(-1/3.0))
                                         * (cos_b1_corrected_flip_angle - 1);
    
        const double r = 1 - E1 * cos_b1_corrected_flip_angle + pown(E2, 2)
                                        * A1 * pow(A2, (double)(1/3.0))
                                        * (cos_b1_corrected_flip_angle - E1);
    
        const double K = (1 - E1 * A1 * cos_b1_corrected_flip_angle
                                  - pown(E2, 2) * pown(A1, 2) * pow(A2, (double)(-2/3.0))
                                                * (E1 * A1 - cos_b1_corrected_flip_angle))
                                    / (E2 * A1 * pow(A2, (double)(-4/3.0))
                                          * (1 + cos_b1_corrected_flip_angle) * (1 - E1 * A1));
    
        const double F1 = K - hypot((double)K, (double)A2);
    
        return -((1 - E1) * E2 * pow(A2, (double)(-2/3.0))
                          * (F1 - E2 * A1 * pow(A2, (double)(2/3.0)))
                          * sin_b1_corrected_flip_angle) / (r - F1*s);
    '''
