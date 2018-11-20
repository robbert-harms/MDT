from mdt import LibraryFunctionTemplate

__author__ = 'Robbert Harms'
__date__ = '2018-07-11'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class BesselRoots(LibraryFunctionTemplate):
    """Constant arrays holding some often used Bessel roots."""
    is_function = False
    cl_code = '''
        /** 
           Zeros of integer-order Bessel function derivative J1'(x).
           Generated using the Python code::
           
               from scipy.special import jnp_zeros
               print(', '.join(map(str, jnp_zeros(1, 16))))
         
         */
        __constant int bessel_roots_jnp_length = 16;
        __constant float bessel_roots_jnp[] = {
            1.8411837813406593, 5.3314427735250325, 8.536316366346286,  11.706004902592063, 
            14.863588633909032, 18.015527862681804, 21.16436985918879,  24.311326857210776, 
            27.457050571059245, 30.601922972669094, 33.746182898667385, 36.88998740923681, 
            40.03344405335068,  43.17662896544882,  46.319597561173914, 49.46239113970275
        };
        
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
        __constant int bessel_roots_j3_2_length = 16;
        __constant float bessel_roots_j3_2[] = {
            2.081575977818101, 5.940369990572713, 9.205840142936664, 12.404445021901974, 
            15.579236410387185, 18.742645584774756, 21.89969647949278, 25.052825280992952, 
            28.203361003952356, 31.352091726564478, 34.49951492136695, 37.645960323086385, 
            40.79165523127188, 43.93676147141978, 47.08139741215418, 50.22565164918307
        };
        
    '''
