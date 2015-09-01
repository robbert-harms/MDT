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
model_float cmCylinderGPD(const model_float4 g,
                     const model_float G,
                     const model_float Delta,
                     const model_float delta,
                     const double d,
                     const double theta,
                     const double phi,
                     const double R,
                     global const model_float* const CLJnpZeros,
                     const int CLJnpZerosLength){

    model_float sum = NeumannCylPerpPGSESum(Delta, delta, d, R, CLJnpZeros, CLJnpZerosLength);

    const model_float4 n = (model_float4)(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta), 0.0);
    model_float omega = (G == 0.0) ? M_PI_2 : acos(dot(n, g * G) / (G * length(n)));

    return exp(-2 * GAMMA_H_SQ * pown(G * sin(omega), 2) * sum) *
            exp(-(Delta - (delta/3.0)) * pown(GAMMA_H * delta * G * cos(omega), 2) * d);
}

#endif // DMRICM_CYLINDERGPD_CL