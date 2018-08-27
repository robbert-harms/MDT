from mdt import LibraryFunctionTemplate

__author__ = 'Robbert Harms'
__date__ = '2018-08-27'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class BinghamNormalization_3x3(LibraryFunctionTemplate):
    description = '''
        Computes 1F1(1/2; 3/2; v), the confluent hypergeometric function of the first kind for a 3x3 matrix [1].

        This computes the normalization factor for the Bingham distribution of a 3x3 matrix by means of a saddlepoint
        approximation [2].

        The argument v is a vector of eigenvalues of a 3x3 matrix V for which you want to compute the Bingham 
        normalization factor.

        Args:
            v: the eigenvalues of the matrix

        References:
        [1] Mardia, K.V., Jupp, P.E., 2000. Distributions on spheres. Directional Statistics. 
            John Wiley & Sons, pp. 159–192  
        [2] Kume A, Wood ATA. Saddlepoint approximations for the Bingham and Fisher-Bingham normalising constants. 
            Biometrika. 2005;92(2):465-476. doi:10.1093/biomet/92.2.465.
    '''
    return_type = 'double'
    parameters = [('mot_float_type*', 'v')]
    dependencies = ['solve_cubic_pol_real']
    cl_code = '''
        if(v[0] == 0 && v[1] == 0 && v[2] == 0){
            return 1;
        }

        double coef_roots[4] = {
            2, 
            -2*v[0] - 2*v[1] - 2*v[2] + 3,
            2*v[0]*v[1] + 2*v[0]*v[2] - 2*v[0] + 2*v[1]*v[2] - 2*v[1] - 2*v[2], 
            -2*v[0]*v[1]*v[2] + v[0]*v[1] + v[0]*v[2] + v[1]*v[2]
        };
        int nmr_roots = solve_cubic_pol_real(coef_roots, coef_roots);
        double t = min(coef_roots[0], min(coef_roots[1], coef_roots[2]));

        double prod = 1;
        for(int i = 0; i < 3; i++){
            if(v[i] - t > 0){
                prod *= 1/sqrt(v[i] - t);
            }
        }

        double K2 = 0.5 * (1.0 / pown(v[0]-t, 2) + 1.0 / pown(v[1]-t, 2) + 1.0 / pown(v[2]-t, 2));
        double K3 =       (1.0 / pown(v[0]-t, 3) + 1.0 / pown(v[1]-t, 3) + 1.0 / pown(v[2]-t, 3));
        double K4 = 3   * (1.0 / pown(v[0]-t, 4) + 1.0 / pown(v[1]-t, 4) + 1.0 / pown(v[2]-t, 4));

        double T = (1/8.) * (K4 / pown(K2, 2)) - (5/24.) * (pown(K3, 2)/pown(K2, 3));  
        return M_PI * sqrt(2.0/K2) * prod * exp(-t + T);        
    '''
