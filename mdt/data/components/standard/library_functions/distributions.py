from mdt import LibraryFunctionTemplate

__author__ = 'Robbert Harms'
__date__ = '2018-08-27'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class ConfluentHyperGeometricFirstKind(LibraryFunctionTemplate):
    """Computes 1F1(1/2; 3/2; e), the confluent hypergeometric function of the first kind for a 3x3 matrix [1].

    This can be used to compute the normalization factor of the Bingham distribution for a 3x3 matrix.
    This implementation uses a saddlepoint approximation [2].

    Args:
        e0, e1, e2: the eigenvalues of the 3x3 matrix for which you want to compute the Bingham
            normalization factor.

    References:
    [1] Mardia, K.V., Jupp, P.E., 2000. Distributions on spheres. Directional Statistics.
        John Wiley & Sons, pp. 159–192
    [2] Kume A, Wood ATA. Saddlepoint approximations for the Bingham and Fisher-Bingham normalising constants.
        Biometrika. 2005;92(2):465-476. doi:10.1093/biomet/92.2.465.
    """
    return_type = 'double'
    parameters = ['e0', 'e1', 'e2']
    dependencies = ['solve_cubic_pol_real']
    cl_code = '''
        if(e0 == 0 && e1 == 0 && e2 == 0){
            return 1;
        }
        
        /** 
            These coefficients are calculated using the sympy code:
            
                x1, x2, x3, t = symbols('x1, x2, x3, t', real=True)
                K1 = cancel((Integer(1)/2 * 1/(x1 - t)) + (Integer(1)/2 * 1/(x2 - t)) + (Integer(1)/2 * 1/(x3 - t)))
                n, d = fraction(K1)
                f = collect(n - d, t)
                print(f)
                
            That is, we create a single polynomial and set the numerator equal to the denominator, such that
            the polynomial equals 1, as in the paper.
        */   
        double coef_roots[4] = {
            2, 
            -2*e0 - 2*e1 - 2*e2 + 3,
            2*e0*e1 + 2*e0*e2 - 2*e0 + 2*e1*e2 - 2*e1 - 2*e2, 
            -2*e0*e1*e2 + e0*e1 + e0*e2 + e1*e2
        };
        int nmr_real_roots = solve_cubic_pol_real(coef_roots, coef_roots);
        
        double t = coef_roots[0];
        if(nmr_real_roots == 2){
            t = min(coef_roots[0], coef_roots[1]);
        }
        else if(nmr_real_roots == 3){
            t = min(coef_roots[0], min(coef_roots[1], coef_roots[2]));
        }

        double prod = 1;
        if(e0 - t > 0){
            prod *= 1/sqrt(e0 - t);
        }
        if(e1 - t > 0){
            prod *= 1/sqrt(e1 - t);
        }
        if(e2 - t > 0){
            prod *= 1/sqrt(e2 - t);
        }

        double K2 = 0.5 * (1.0 / pown(e0-t, 2) + 1.0 / pown(e1-t, 2) + 1.0 / pown(e2-t, 2));
        double K3 =       (1.0 / pown(e0-t, 3) + 1.0 / pown(e1-t, 3) + 1.0 / pown(e2-t, 3));
        double K4 = 3   * (1.0 / pown(e0-t, 4) + 1.0 / pown(e1-t, 4) + 1.0 / pown(e2-t, 4));

        double T = (1/8.) * (K4 / pown(K2, 2)) - (5/24.) * (pown(K3, 2)/pown(K2, 3));  
        return M_PI * sqrt(2.0 / K2) * prod * exp(-t + T);        
    '''
