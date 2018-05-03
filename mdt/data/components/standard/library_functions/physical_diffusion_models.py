from mdt import LibraryFunctionTemplate

__author__ = 'Robbert Harms'
__date__ = '2018-05-02'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class NeumanCylinder(LibraryFunctionTemplate):
    description = '''
        This function returns the displacement in the restricted signal attenuation, for radius R 
        according to the C.H. Neuman model, equation (20) in [1].

        In spin-echo experiments, tau is typically TE/2.

        References:
        1) 1. Neuman CH. Spin echo of spins diffusing in a bounded medium. 
            J Chem Phys. 1974;60(11):4508-4511. doi:10.1063/1.1680931.
    '''
    return_type = 'double'
    parameters = ['G', 'tau', 'd', 'R']
    dependencies = ['MRIConstants']
    cl_code = '''
        if(R == 0.0 || R < MOT_EPSILON){
            return 0;
        }

        const mot_float_type cl_jnp_zeros[] = {
            1.8411837813406593, 5.3314427735250325, 8.536316366346286,  11.706004902592063, 
            14.863588633909032, 18.015527862681804, 21.16436985918879,  24.311326857210776, 
            27.457050571059245, 30.601922972669094, 33.746182898667385, 36.88998740923681, 
            40.03344405335068,  43.17662896544882,  46.319597561173914, 49.46239113970275
        };
        const int cl_jnp_zeros_length = 16;

        double sum = 0;
        mot_float_type alpha;
        
        #pragma unroll
        for(uint i = 0; i < cl_jnp_zeros_length; i++){
            alpha = cl_jnp_zeros[i] / R;

            sum += (pown(alpha, -4) / (cl_jnp_zeros[i] * cl_jnp_zeros[i] - 1)) * 
                    (2 * tau - 
                        (3 - 4 * exp(-alpha * alpha * d * tau) + exp(- alpha * alpha * d * 2 * tau)) 
                         / (alpha * alpha * d)
                     );
        }        

        return -((2 * GAMMA_H_SQ * (G*G)) / d) * sum;
    '''


class NeumanCylinderLongApprox(LibraryFunctionTemplate):
    description = '''
        This function returns the displacement in the restricted signal attenuation, for radius R according to the 
        long diffusion time approximation of the C.H. Neuman Neumann model, equation (28) in [1].

        In typical spin-echo experiments, ``tau = TE/2``.

        References:
        1) 1. Neuman CH. Spin echo of spins diffusing in a bounded medium. 
            J Chem Phys. 1974;60(11):4508-4511. doi:10.1063/1.1680931.
    '''
    return_type = 'double'
    parameters = ['G', 'tau', 'd', 'R']
    dependencies = ['MRIConstants']
    cl_code = '''
        return -( (pown(R, 4) * GAMMA_H_HZ_SQ * (G * G)) / d ) 
                    * (7 / 296.0) 
                    * (2 * tau - (99/112.0) * (pown(R, 2) / d)); 
    '''


class VanGelderenCylinder(LibraryFunctionTemplate):
    description = '''
        This function returns the displacement in the restricted signal attenuation for radius R 
        according to the Van Gelderen model [1].

        References:
        1) Gelderen V, D D, PC van Z, CT M. Evaluation of Restricted Diffusion in Cylinders. 
            Phosphocreatine in Rabbit Leg Muscle. 1994. doi:10.1006/jmrb.1994.1038.
    '''
    return_type = 'double'
    parameters = ['G', 'Delta', 'delta', 'd', 'R']
    dependencies = ['MRIConstants']
    cl_code = '''
        if(R == 0.0 || R < MOT_EPSILON){
            return 0;
        }

        const mot_float_type cl_jnp_zeros[] = {
            1.8411837813406593, 5.3314427735250325, 8.536316366346286,  11.706004902592063, 
            14.863588633909032, 18.015527862681804, 21.16436985918879,  24.311326857210776, 
            27.457050571059245, 30.601922972669094, 33.746182898667385, 36.88998740923681, 
            40.03344405335068,  43.17662896544882,  46.319597561173914, 49.46239113970275
        };
        const int cl_jnp_zeros_length = 16;

        double sum = 0;
        mot_float_type alpha;
        mot_float_type alpha2_d;
        
        #pragma unroll
        for(uint i = 0; i < cl_jnp_zeros_length; i++){
            alpha = cl_jnp_zeros[i] / R;
            alpha2_d = d * alpha * alpha;

            sum += (2 * alpha2_d * delta
                    -  2
                    + (2 * exp(-alpha2_d * delta))
                    + (2 * exp(-alpha2_d * Delta))
                    - exp(-alpha2_d * (Delta - delta))
                    - exp(-alpha2_d * (Delta + delta)))
                        / ((alpha2_d * alpha * alpha2_d * alpha) * (cl_jnp_zeros[i] * cl_jnp_zeros[i] - 1));
        }
        return -2 * GAMMA_H_SQ * (G*G) * sum;
    '''
