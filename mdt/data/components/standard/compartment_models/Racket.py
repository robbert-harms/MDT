from mdt import CompartmentTemplate
import numpy as np

__author__ = 'Robbert Harms'
__date__ = '2018-08-27'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class Racket(CompartmentTemplate):
    """Computes the Racket model from [1].

    As in the original paper, we optimize "kw = k1/k2"  to preserve the inequality k1 >= k2 during optimization.

    References:
    [1] Sotiropoulos SN, Behrens TEJ, Jbabdi S. Ball and rackets: Inferring fiber fanning from
        diffusion-weighted MRI. Neuroimage. 2012;60(2):1412-1425. doi:10.1016/j.neuroimage.2012.01.056.
    """
    parameters = ('g', 'b', 'd', 'theta', 'phi', 'psi', 'k1', 'kw', '@cache')
    dependencies = ['eigenvalues_3x3_symmetric', 'ConfluentHyperGeometricFirstKind']
    cl_code = '''
        /**
         *  This computes Q = R.T*B_diag*R * -bdg^2 in one go. 
         
            This code was generated using sympy and some manual adjustments::

                from sympy import symbols, Matrix, simplify, cos, sin
                psi, phi, theta, k1, k2 = symbols("psi, phi, theta, k1, k2")
    
                R_psi = Matrix([[cos(psi), sin(psi), 0], [-sin(psi), cos(psi), 0], [0, 0, 1]])
                R_theta = Matrix([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]])
                R_phi = Matrix([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]])
                B_diag = Matrix([[-k1, 0, 0], [0, -k2, 0], [0, 0, 0]])
    
                R = R_psi * R_theta * R_phi
                B = simplify(R.T * B_diag * R)

            Please note that Q = B-bdg^2 is supposed to be symmetric.
        */
        double Q[6]; // upper triangular
        Q[0] = -b*d*g.x*g.x + cache->B[0];
        Q[1] = -b*d*g.x*g.y + cache->B[1];
        Q[2] = -b*d*g.x*g.z + cache->B[2];
        Q[3] = -b*d*g.y*g.y + cache->B[3];
        Q[4] = -b*d*g.y*g.z + cache->B[4];
        Q[5] = -b*d*g.z*g.z + cache->B[5];

        double e[3];
        eigenvalues_3x3_symmetric(Q, e);
        
        return ConfluentHyperGeometricFirstKind(-e[0], -e[1], -e[2]) / *cache->denom;
    '''
    cache_info = {
        'fields': ['double denom', ('double', 'B', 6)],
        'cl_code': '''
            double k2 = k1 / kw;
            
            double cos_theta;
            double sin_theta = sincos(theta, &cos_theta);
            double cos_phi;
            double sin_phi = sincos(phi, &cos_phi);
            double cos_psi;
            double sin_psi = sincos(psi, &cos_psi);
            
            cache->B[0] = -k1*pown(sin_phi*sin_psi - cos_phi*cos_psi*cos_theta, 2) 
                            - k2*pown(sin_phi*cos_psi + sin_psi*cos_phi*cos_theta, 2);
            cache->B[1] = k1*(sin_phi*sin_psi - cos_phi*cos_psi*cos_theta)*(sin_phi*cos_psi*cos_theta + sin_psi*cos_phi) 
                            - k2*(sin_phi*cos_psi + sin_psi*cos_phi*cos_theta)
                                    *(sin_phi*sin_psi*cos_theta - cos_phi*cos_psi);
            cache->B[2] = (-k1*(sin_phi*sin_psi - cos_phi*cos_psi*cos_theta)*cos_psi 
                            + k2*(sin_phi*cos_psi + sin_psi*cos_phi*cos_theta)*sin_psi)*sin_theta;
            cache->B[3] = -k1*pown(sin_phi*cos_psi*cos_theta + sin_psi*cos_phi, 2) 
                            - k2*pown(sin_phi*sin_psi*cos_theta - cos_phi*cos_psi, 2);
            cache->B[4] = (k1*(sin_phi*cos_psi*cos_theta + sin_psi*cos_phi)*cos_psi 
                            + k2*(sin_phi*sin_psi*cos_theta - cos_phi*cos_psi)*sin_psi)*sin_theta;
            cache->B[5] = -(k1*cos_psi*cos_psi + k2*sin_psi*sin_psi)*sin_theta*sin_theta;
            
            *cache->denom = ConfluentHyperGeometricFirstKind(k1, k2, 0);
        '''
    }
    extra_optimization_maps = [
        lambda d: {'k2': d['k1'] / d['kw']},
        lambda d: {'ODI_k1': np.arctan2(1.0, d['k1']) * 2 / np.pi,
                   'ODI_k2': np.arctan2(1.0, d['k2']) * 2 / np.pi,
                   'ODI': np.sqrt(np.abs(np.arctan2(1.0, (d['k1'] * d['k2'])) * 2 / np.pi))}
    ]
    extra_sampling_maps = [
        lambda samples: {'k1': np.mean(samples['k1'], axis=1),
                         'k1.std': np.std(samples['k1'], axis=1),
                         'k2': np.mean(samples['k1'] / samples['kw'], axis=1),
                         'k2.std': np.std(samples['k1'] / samples['kw'], axis=1)}
    ]
