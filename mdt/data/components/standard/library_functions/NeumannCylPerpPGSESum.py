from mdt.components_config.library_functions import LibraryFunctionConfig


__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NeumannCylPerpPGSESum(LibraryFunctionConfig):

    description = '''
        This function returns the summation of the signal attenuation in perpendicular direction (LePerp)
        for Radius R, according to the Neumann model.

        The summation is the sum over the Bessel roots up to a accuracy of 1e-8, it does not
        calculate the complete signal for a cylinder dMRI compartment model.
    '''
    return_type = 'double'
    parameter_list = ['Delta', 'delta', 'd', 'R']
    cl_code = '''
        if(R == 0.0 || R < MOT_EPSILON){
            return 0;
        }

        const mot_float_type cl_jnp_zeros[] = {
            1.84118378,   5.33144277,   8.53631637,  11.7060049 ,
            14.86358863,  18.01552786,  21.16436986,  24.31132686,
            27.45705057,  30.60192297,  33.7461829 ,  36.88998741,
            40.03344405,  43.17662897,  46.31959756,  49.46239114,
            52.60504111,  55.74757179,  58.8900023 ,  62.03234787
        };
        const int cl_jnp_zeros_length = 20;

        double sum = 0;
        mot_float_type dam;
        mot_float_type amrdiv;

        for(int i = 0; i < cl_jnp_zeros_length; i++){
            amrdiv = cl_jnp_zeros[i] / R;
            dam = d * amrdiv * amrdiv;

            sum += (2 * dam * delta
                        -  2
                        + (2 * exp(-dam * delta))
                        + (2 * exp(-dam * Delta))
                        - exp(-dam * (Delta - delta))
                        - exp(-dam * (Delta + delta)))
                    / ((dam * amrdiv * dam * amrdiv) * ((R * amrdiv * R * amrdiv) - 1));
        }
        return sum;
    '''
