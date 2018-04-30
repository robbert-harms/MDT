from mdt.component_templates.library_functions import LibraryFunctionTemplate


__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NeumannCylinderRestricted(LibraryFunctionTemplate):

    description = '''
        This function returns the displacement in the restricted signal attenuation, for radius R 
        according to the Von Neumann model, equation (20) in [1].
        
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
            40.03344405335068,  43.17662896544882,  46.319597561173914, 49.46239113970275, 
            52.60504111155669,  55.74757179225101,  58.8900022991857,   62.03234787066199
        };
        const int cl_jnp_zeros_length = 20;
        
        double sum = 0;
        mot_float_type alpha;
        
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
