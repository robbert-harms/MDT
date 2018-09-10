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
        __constant mot_float_type bessel_roots_jnp[] = {
            1.8411837813406593, 5.3314427735250325, 8.536316366346286,  11.706004902592063, 
            14.863588633909032, 18.015527862681804, 21.16436985918879,  24.311326857210776, 
            27.457050571059245, 30.601922972669094, 33.746182898667385, 36.88998740923681, 
            40.03344405335068,  43.17662896544882,  46.319597561173914, 49.46239113970275
        };
    '''
