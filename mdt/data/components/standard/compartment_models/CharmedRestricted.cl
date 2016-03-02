/**
 * Author = Robbert Harms
 * Date = 2014-11-06
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

mot_float_type cmCharmedRestricted(const mot_float_type4 g,
                                   const mot_float_type b,
                                   const mot_float_type q,
                                   const mot_float_type Delta,
                                   const mot_float_type delta,
                                   const mot_float_type TE,
                                   const mot_float_type d,
                                   const mot_float_type theta,
                                   const mot_float_type phi){

    const double direction = pown(dot(g, (mot_float_type4)(cos(phi) * sin(theta),
                                        sin(phi) * sin(theta), cos(theta), 0.0)), 2);

    const double signal_par = -(4 * (M_PI * M_PI) * (q*q) * direction * (Delta - (delta / 3.0)) * d);
    const double signal_perp_tmp1 = -( (4 * (M_PI * M_PI) * (1 - direction) * (7/96.0)) / (d * (TE / 2.0)) );
    const double signal_perp_tmp2 = (99/112.0) / (d * (TE / 2.0));

    double sum = 0.0;
    // R is the radius of the cylinder, this is in units of meters.
    //     cylinder_weight                                          R^4                                        R^2
    sum += 0.021184720085574 * exp(signal_par + (signal_perp_tmp1 * 5.0625e-24      * (2 - (signal_perp_tmp2 * 2.25e-12))));
    sum += 0.107169623942214 * exp(signal_par + (signal_perp_tmp1 * 3.90625e-23     * (2 - (signal_perp_tmp2 * 6.25e-12))));
    sum += 0.194400551313197 * exp(signal_par + (signal_perp_tmp1 * 1.500625e-22    * (2 - (signal_perp_tmp2 * 1.225e-11))));
    sum += 0.266676876170322 * exp(signal_par + (signal_perp_tmp1 * 4.100625e-22    * (2 - (signal_perp_tmp2 * 2.025e-11))));
    sum += 0.214921653661151 * exp(signal_par + (signal_perp_tmp1 * 9.150625e-22    * (2 - (signal_perp_tmp2 * 3.025e-11))));
    sum += 0.195646574827541 * exp(signal_par + (signal_perp_tmp1 * 1.785061655e-21 * (2 - (signal_perp_tmp2 * 4.224999e-11))));

    return (mot_float_type)sum;
}


