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

    const double dotted = pown(dot(g, (MOT_FLOAT_TYPE4)(cos(phi) * sin(theta),
                                        sin(phi) * sin(theta), cos(theta), 0.0)), 2);
    const double tmp1 = (-d * b * dotted);
    const double tmp2 = ((-GAMMA2_G2_delta2 * (1 - dotted)) / (d * (TE / 2.0))) * (7/96.0);
    const double tmp3 = (99/112.0) / (d * (TE / 2.0));

    // written out version of:
    //sum += gamma_cyl_weights[i] *
    //            exp(tmp1 + (tmp2 * pown(gamma_cyl_radii_sq[i], 2) * (2 - (tmp3 * gamma_cyl_radii_sq[i]))));
    // where gamma_cyl_weights is an array containing the cylinder weights
    // and gamma_cyl_radii_sq containing the radius per cylinder.

    double sum = 0.0;
    sum += 0.0211847200855742 * exp(tmp1 + (tmp2 * 5.0625e-24 * (2 - (tmp3 * 2.25e-12))));
    sum += 0.107169623942214 * exp(tmp1 + (tmp2 * 3.90625e-23 * (2 - (tmp3 * 6.25e-12))));
    sum += 0.194400551313197 * exp(tmp1 + (tmp2 * 1.500625e-22 * (2 - (tmp3 * 1.225e-11))));
    sum += 0.266676876170322 * exp(tmp1 + (tmp2 * 4.100625e-22 * (2 - (tmp3 * 2.025e-11))));
    sum += 0.214921653661151 * exp(tmp1 + (tmp2 * 9.150625e-22 * (2 - (tmp3 * 3.025e-11))));
    sum += 0.195646574827541 * exp(tmp1 + (tmp2 * 1.785061655e-21 * (2 - (tmp3 * 4.224999e-11))));

    return (MOT_FLOAT_TYPE)sum;
}


