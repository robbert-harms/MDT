/**
 * Author = Robbert Harms
 * Date = 2014-11-06
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

MOT_FLOAT_TYPE cmCharmedRestricted(const MOT_FLOAT_TYPE4 g,
                                   const MOT_FLOAT_TYPE b,
                                   const MOT_FLOAT_TYPE GAMMA2_G2_delta2,
                                   const MOT_FLOAT_TYPE TE,
                                   const MOT_FLOAT_TYPE d,
                                   const MOT_FLOAT_TYPE theta,
                                   const MOT_FLOAT_TYPE phi){

    // gamma_cyl_radii squared
    const double gamma_cyl_radii_sq[] = {
        2.25e-12, 6.25e-12, 1.225e-11,
        2.025e-11, 3.025e-11, 4.224999e-11
    };

    const double gamma_cyl_weights[] = {
        0.0211847200855742, 0.107169623942214, 0.194400551313197,
        0.266676876170322, 0.214921653661151, 0.195646574827541};

    const int gamma_cyl_length = 6;

    const double dotted = pown(dot(g, (MOT_FLOAT_TYPE4)(cos(phi) * sin(theta),
                                        sin(phi) * sin(theta), cos(theta), 0.0)), 2);
    const double tmp1 = (-d * b * dotted);
    const double tmp2 = ((-GAMMA2_G2_delta2 * (1 - dotted)) / (d * (TE / 2.0))) * (7/96.0);
    const double tmp3 = (99/112.0) / (d * (TE / 2.0));

    double sum = 0;
    for(int i = 0; i < gamma_cyl_length; i++){
        sum += gamma_cyl_weights[i] * exp(tmp1 + (tmp2 * pown(gamma_cyl_radii_sq[i], 2) *
                                                    (2 - (tmp3 * gamma_cyl_radii_sq[i]))));
    }
    return (MOT_FLOAT_TYPE)sum;
}


