#ifndef DMRICM_CYLINDERGPD_CL
#define DMRICM_CYLINDERGPD_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Generate the compartment model signal for the CylinderGPD model.
 */
MOT_FLOAT_TYPE cmCylinderGPD(const MOT_FLOAT_TYPE4 g,
                          const MOT_FLOAT_TYPE G,
                          const MOT_FLOAT_TYPE Delta,
                          const MOT_FLOAT_TYPE delta,
                          const MOT_FLOAT_TYPE d,
                          const MOT_FLOAT_TYPE theta,
                          const MOT_FLOAT_TYPE phi,
                          const MOT_FLOAT_TYPE R){

    MOT_FLOAT_TYPE sum = NeumannCylPerpPGSESum(Delta, delta, d, R);

    const MOT_FLOAT_TYPE4 n = (MOT_FLOAT_TYPE4)(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta), 0.0);
    MOT_FLOAT_TYPE omega = (G == 0.0) ? M_PI_2 : acos(dot(n, g * G) / (G * length(n)));

    return exp(-2 * GAMMA_H_SQ * pown(G * sin(omega), 2) * sum) *
            exp(-(Delta - (delta/3.0)) * pown(GAMMA_H * delta * G * cos(omega), 2) * d);
}

#endif // DMRICM_CYLINDERGPD_CL