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
model_float cmCylinderGPD(const double4 g,
                     const double G,
                     const double Delta,
                     const double delta,
                     const double d,
                     const double theta,
                     const double phi,
                     const double R,
                     global const double* const CLJnpZeros,
                     const int CLJnpZerosLength){

    double sum = NeumannCylPerpPGSESum(Delta, delta, d, R, CLJnpZeros, CLJnpZerosLength);

    const double4 n = (double4)(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta), 0.0);
    double omega = (G == 0.0) ? M_PI_2 : acos(dot(n, g * G) / (G * length(n)));

    return exp(-2 * GAMMA_H_SQ * pown(G * sin(omega), 2) * sum) *
            exp(-(Delta - (delta/3.0)) * pown(GAMMA_H * delta * G * cos(omega), 2) * d);
}

#endif // DMRICM_CYLINDERGPD_CL