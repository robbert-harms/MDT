from mdt import LibraryFunctionTemplate

__author__ = 'Robbert Harms'
__date__ = '2018-05-02'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class NeumanCylinder(LibraryFunctionTemplate):
    """This function returns the displacement in the restricted signal attenuation, for radius R
        according to the C.H. Neuman model, equation (20) in [1].

    In spin-echo experiments, tau is typically TE/2.

    References:
    [1] Neuman CH. Spin echo of spins diffusing in a bounded medium.
        J Chem Phys. 1974;60(11):4508-4511. doi:10.1063/1.1680931.
    """
    return_type = 'double'
    parameters = ['G', 'tau', 'd', 'R']
    dependencies = ['MRIConstants', 'BesselRoots']
    cl_code = '''
        if(R == 0.0 || R < MOT_EPSILON){
            return 0;
        }

        double sum = 0;
        mot_float_type alpha;
        
        #pragma unroll
        for(uint i = 0; i < bessel_roots_jnp_length; i++){
            alpha = bessel_roots_jnp[i] / R;

            sum += (pown(alpha, -4) / (bessel_roots_jnp[i] * bessel_roots_jnp[i] - 1)) * 
                    (2 * tau - 
                        (3 - 4 * exp(-alpha * alpha * d * tau) + exp(- alpha * alpha * d * 2 * tau)) 
                         / (alpha * alpha * d)
                     );
        }        

        return -((2 * GAMMA_H_SQ * (G*G)) / d) * sum;
    '''


class NeumanSphere(LibraryFunctionTemplate):
    """This function returns the displacement in spherical signal attenuation, for radius R
        according to the C.H. Neuman model, equation (18) in [1].

    In spin-echo experiments, tau is typically TE/2.

    References:
    [1] Neuman CH. Spin echo of spins diffusing in a bounded medium.
        J Chem Phys. 1974;60(11):4508-4511. doi:10.1063/1.1680931.
    """
    return_type = 'double'
    parameters = ['G', 'tau', 'd', 'R']
    dependencies = ['MRIConstants']
    cl_code = '''
        if(R == 0.0 || R < MOT_EPSILON){
            return 0;
        }

        /**
         *  Zeros of (am*x)j3/2'(am*x)- 1/2 J3/2(am*x), used in computing diffusion in spherical boundaries
            using the Neuman approximation.

            Computed using the Python code:

            from mpmath import findroot
            from scipy.special import jvp, jv
            import numpy as np


            def f(a):
                """Computes (am*x)j3/2'(am*x)- 1/2 J3/2(am*x)=0

                References:
                [1] Neuman CH. Spin echo of spins diffusing in a bounded medium. J Chem Phys.
                    1974;60(11):4508-4511. doi:10.1063/1.1680931.
                """
                a = float(a)
                return a * jvp(3/2., a) - 0.5 * jv(3/2., a)


            print(', '.join([str(float(findroot(f, k))) for k in np.arange(2, 16*np.pi, np.pi)]))
        */
        const mot_float_type am_zeros[] = {
            2.081575977818101, 5.940369990572713, 9.205840142936664, 12.404445021901974, 
            15.579236410387185, 18.742645584774756, 21.89969647949278, 25.052825280992952, 
            28.203361003952356, 31.352091726564478, 34.49951492136695, 37.645960323086385, 
            40.79165523127188, 43.93676147141978, 47.08139741215418, 50.22565164918307
        };
        const int am_zeros_length = 16;

        double sum = 0;
        mot_float_type alpha;

        #pragma unroll
        for(uint i = 0; i < am_zeros_length; i++){
            alpha = am_zeros[i] / R;

            sum += (pown(alpha, -4) / (am_zeros[i] * am_zeros[i] - 2)) * 
                    (2 * tau - 
                        (3 - 4 * exp(-alpha * alpha * d * tau) + exp(- alpha * alpha * d * 2 * tau)) 
                         / (alpha * alpha * d)
                     );
        }        

        return -((2 * GAMMA_H_SQ * (G*G)) / d) * sum;
    '''


class NeumanCylinderLongApprox(LibraryFunctionTemplate):
    """This function returns the displacement in the restricted signal attenuation, for radius R according to the
       long diffusion time approximation of the C.H. Neuman model, equation (28) in [1].

    In typical spin-echo experiments, ``tau = TE/2``.

    References:
    [1] Neuman CH. Spin echo of spins diffusing in a bounded medium.
        J Chem Phys. 1974;60(11):4508-4511. doi:10.1063/1.1680931.
    """
    return_type = 'double'
    parameters = ['G', 'tau', 'd', 'R']
    dependencies = ['MRIConstants']
    cl_code = '''
        return -( (pown(R, 4) * GAMMA_H_HZ_SQ * (G * G)) / d ) 
                    * (7 / 296.0) 
                    * (2 * tau - (99/112.0) * (pown(R, 2) / d)); 
    '''


class NeumanSphereLongApprox(LibraryFunctionTemplate):
    """This function returns the displacement in the spherical signal attenuation, for radius R according to the
       long diffusion time approximation of the C.H. Neuman model, equation (27) in [1].

    In typical spin-echo experiments, ``tau = TE/2``.

    References:
    [1] Neuman CH. Spin echo of spins diffusing in a bounded medium.
        J Chem Phys. 1974;60(11):4508-4511. doi:10.1063/1.1680931.
    """
    return_type = 'double'
    parameters = ['G', 'tau', 'd', 'R']
    dependencies = ['MRIConstants']
    cl_code = '''
        return -( (pown(R, 4) * GAMMA_H_HZ_SQ * (G * G)) / d ) 
                    * (8 / 175.0) 
                    * (2 * tau - (581/840.0) * (pown(R, 2) / d)); 
    '''


class VanGelderenCylinder(LibraryFunctionTemplate):
    """This function returns the displacement in the restricted signal attenuation for radius R
       according to the Van Gelderen model [1, 2].

    References:
    [1] Gelderen V, D D, PC van Z, CT M. Evaluation of Restricted Diffusion in Cylinders.
        Phosphocreatine in Rabbit Leg Muscle. 1994. doi:10.1006/jmrb.1994.1038.
    [2] 1. Wang LZ, Caprihan A, Fukushima E. The narrow-pulse criterion for pulsed-gradient spin-echo
        diffusion measurements. JMagnResonSerA. 1995;117(2):209-219.
    """
    return_type = 'double'
    parameters = ['G', 'Delta', 'delta', 'd', 'R']
    dependencies = ['MRIConstants', 'BesselRoots']
    cl_code = '''
        if(R == 0.0 || R < MOT_EPSILON){
            return 0;
        }

        double sum = 0;
        mot_float_type alpha;
        mot_float_type alpha2_d;
        
        #pragma unroll
        for(uint i = 0; i < bessel_roots_jnp_length; i++){
            alpha = bessel_roots_jnp[i] / R;
            alpha2_d = d * alpha * alpha;

            sum += (2 * alpha2_d * delta
                    -  2
                    + (2 * exp(-alpha2_d * delta))
                    + (2 * exp(-alpha2_d * Delta))
                    - exp(-alpha2_d * (Delta - delta))
                    - exp(-alpha2_d * (Delta + delta)))
                        / ((alpha2_d * alpha * alpha2_d * alpha) * (bessel_roots_jnp[i] * bessel_roots_jnp[i] - 1));
        }
        return -2 * GAMMA_H_SQ * (G*G) * sum;
    '''


class VanGelderenSphere(LibraryFunctionTemplate):
    """This function returns the displacement in the spherical signal attenuation for radius R
       according to the Van Gelderen model [1].

    References:
    [1] Wang LZ, Caprihan A, Fukushima E. The narrow-pulse criterion for pulsed-gradient spin-echo
        diffusion measurements. JMagnResonSerA. 1995;117(2):209-219.
    """
    return_type = 'double'
    parameters = ['G', 'Delta', 'delta', 'd', 'R']
    dependencies = ['MRIConstants']
    cl_code = '''
        if(R == 0.0 || R < MOT_EPSILON){
            return 0;
        }
        
        const mot_float_type am_zeros[] = {
            2.081575977818101, 5.940369990572713, 9.205840142936664, 12.404445021901974, 
            15.579236410387185, 18.742645584774756, 21.89969647949278, 25.052825280992952, 
            28.203361003952356, 31.352091726564478, 34.49951492136695, 37.645960323086385, 
            40.79165523127188, 43.93676147141978, 47.08139741215418, 50.22565164918307
        };
        const int am_zeros_length = 16;
        
        double sum = 0;
        mot_float_type alpha;
        mot_float_type alpha2_d;
        
        #pragma unroll
        for(uint i = 0; i < am_zeros_length; i++){
            alpha = am_zeros[i] / R;
            alpha2_d = d * alpha * alpha;

            sum += (2 * alpha2_d * delta
                    -  2
                    + (2 * exp(-alpha2_d * delta))
                    + (2 * exp(-alpha2_d * Delta))
                    - exp(-alpha2_d * (Delta - delta))
                    - exp(-alpha2_d * (Delta + delta)))
                        / ((alpha2_d * alpha * alpha2_d * alpha) * (am_zeros[i] * am_zeros[i] - 2));
        }
        return -2 * GAMMA_H_SQ * (G*G) * sum;
    '''
