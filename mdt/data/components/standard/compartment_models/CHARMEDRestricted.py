from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CHARMEDRestricted(CompartmentTemplate):

    parameter_list = ('g', 'b', 'G', 'Delta', 'delta', 'TE', 'd', 'theta', 'phi')
    dependency_list = ('MRIConstants', 'SphericalToCartesian')
    cl_code = '''
        const mot_float_type q_magnitude_2 = GAMMA_H_HZ_SQ * (G * G) * (delta * delta);
        const mot_float_type direction_2 = pown(dot(g, SphericalToCartesian(theta, phi)), 2);

        const mot_float_type signal_par = -(4 * (M_PI_F * M_PI_F) * q_magnitude_2 * direction_2 * (Delta - (delta / 3.0)) * d);
        const mot_float_type signal_perp_tmp1 = -( (4 * (M_PI_F * M_PI_F) * q_magnitude_2 * (1 - direction_2) * (7/96.0)) / (d * (TE / 2.0)));
        const mot_float_type signal_perp_tmp2 = (99/112.0) / (d * (TE / 2.0));

        // R is the radius of the cylinder in meters
        //      cylinder_weight                                          R^4                                        R^2
        return (0.021184720085574 * exp(signal_par + (signal_perp_tmp1 * 5.0625e-24      * (2 - (signal_perp_tmp2 * 2.25e-12))))) +
               (0.107169623942214 * exp(signal_par + (signal_perp_tmp1 * 3.90625e-23     * (2 - (signal_perp_tmp2 * 6.25e-12))))) +
               (0.194400551313197 * exp(signal_par + (signal_perp_tmp1 * 1.500625e-22    * (2 - (signal_perp_tmp2 * 1.225e-11))))) +
               (0.266676876170322 * exp(signal_par + (signal_perp_tmp1 * 4.100625e-22    * (2 - (signal_perp_tmp2 * 2.025e-11))))) +
               (0.214921653661151 * exp(signal_par + (signal_perp_tmp1 * 9.150625e-22    * (2 - (signal_perp_tmp2 * 3.025e-11))))) +
               (0.195646574827541 * exp(signal_par + (signal_perp_tmp1 * 1.785061655e-21 * (2 - (signal_perp_tmp2 * 4.224999e-11)))));
    '''
