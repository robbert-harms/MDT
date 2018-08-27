from mdt import FreeParameterTemplate, CompartmentTemplate
import numpy as np

__author__ = 'Robbert Harms'
__date__ = '2018-08-27'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class Racket(CompartmentTemplate):
    description = '''
        Computes the Racket model from [1].

        As in the original paper, we optimize "kw = k1/k2"  to preserve the inequality k1 >= k2 during optimization.

        References:
        [1] Sotiropoulos SN, Behrens TEJ, Jbabdi S. Ball and rackets: Inferring fiber fanning from 
            diffusion-weighted MRI. Neuroimage. 2012;60(2):1412-1425. doi:10.1016/j.neuroimage.2012.01.056.
    '''
    parameters = ('g', 'b', 'd', 'theta', 'phi', 'psi', 'k1', 'kw')
    dependencies = ['EigenvaluesSymmetric3x3', 'BinghamNormalization_3x3']
    cl_code = '''
        double k2 = k1 / kw;

        mot_float_type cos_theta;
        mot_float_type sin_theta = sincos(theta, &cos_theta);
        mot_float_type cos_phi;
        mot_float_type sin_phi = sincos(phi, &cos_phi);
        mot_float_type cos_psi;
        mot_float_type sin_psi = sincos(psi, &cos_psi);

        /**
         * This computes R.T*B_diag*R * -bdg^2 in one go. This code was generated using sympy and some manual adjustments:

            from sympy import symbols, Matrix, simplify, cos, sin
            psi, phi, theta, k1, k2 = symbols("psi, phi, theta, k1, k2")

            R_psi = Matrix([[cos(psi), sin(psi), 0], [-sin(psi), cos(psi), 0], [0, 0, 1]])
            R_theta = Matrix([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]])
            R_phi = Matrix([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]])
            B_diag = Matrix([[-k1, 0, 0], [0, -k2, 0], [0, 0, 0]])

            R = R_psi * R_theta * R_phi
            B = simplify(R.T * B_diag * R)

            Please note that B is symmetric.
        */
        mot_float_type B[9];
        B[0] = -b*d*g.x*g.x + -k1*pown(sin_phi*sin_psi - cos_phi*cos_psi*cos_theta, 2) - k2*pown(sin_phi*cos_psi + sin_psi*cos_phi*cos_theta, 2);
        B[1] = -b*d*g.x*g.y + k1*(sin_phi*sin_psi - cos_phi*cos_psi*cos_theta)*(sin_phi*cos_psi*cos_theta + sin_psi*cos_phi) - k2*(sin_phi*cos_psi + sin_psi*cos_phi*cos_theta)*(sin_phi*sin_psi*cos_theta - cos_phi*cos_psi);
        B[2] = -b*d*g.x*g.z + (-k1*(sin_phi*sin_psi - cos_phi*cos_psi*cos_theta)*cos_psi + k2*(sin_phi*cos_psi + sin_psi*cos_phi*cos_theta)*sin_psi)*sin_theta;
        B[3] = B[1];
        B[4] = -b*d*g.y*g.y + -k1*pown(sin_phi*cos_psi*cos_theta + sin_psi*cos_phi, 2) - k2*pown(sin_phi*sin_psi*cos_theta - cos_phi*cos_psi, 2);
        B[5] = -b*d*g.y*g.z + (k1*(sin_phi*cos_psi*cos_theta + sin_psi*cos_phi)*cos_psi + k2*(sin_phi*sin_psi*cos_theta - cos_phi*cos_psi)*sin_psi)*sin_theta;
        B[6] = B[2];
        B[7] = B[5];
        B[8] = -b*d*g.z*g.z + -(k1*cos_psi*cos_psi + k2*sin_psi*sin_psi)*sin_theta*sin_theta;

        mot_float_type v[3] = {k1, k2, 0};
        double denominator = BinghamNormalization_3x3(v);

        EigenvaluesSymmetric3x3(B, v);
        v[0] *= -1; v[1] *= -1; v[2] *= -1;        
        double numerator = BinghamNormalization_3x3(v);

        return (numerator/denominator);
    '''
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

    class k1(FreeParameterTemplate):
        init_value = 20
        lower_bound = 0
        upper_bound = 64
        parameter_transform = 'CosSqrClamp'
        sampling_proposal_std = 0.1
        numdiff_info = {'max_step': 0.1, 'use_upper_bound': False}

    class kw(FreeParameterTemplate):
        """We optimize the ratio w = k1/k2"""
        init_value = 4
        lower_bound = 1
        upper_bound = 64
        parameter_transform = 'CosSqrClamp'
        sampling_proposal_std = 0.1
        numdiff_info = {'max_step': 0.1, 'use_upper_bound': False}

