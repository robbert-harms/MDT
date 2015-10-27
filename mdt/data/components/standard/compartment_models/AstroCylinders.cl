#ifndef DMRICM_ASTROCYLINDERS_CL
#define DMRICM_ASTROCYLINDERS_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

model_float cmAstroCylinders(const model_float4 g,
                             const model_float b,
                             const model_float G,
                             const model_float Delta,
                             const model_float delta,
                             const model_float d,
                             const model_float R,
                             global const model_float* const CLJnpZeros,
                             const int CLJnpZerosLength){

    model_float sum = NeumannCylPerpPGSESum(Delta, delta, d, R, CLJnpZeros, CLJnpZerosLength);

    model_float lperp = (-2 * GAMMA_H_SQ * sum);
	model_float lpar = -b * 1.0/pown(G, 2) * d;

    return (sqrt(M_PI) / (2 * G * sqrt(lperp - lpar)))
			    * exp(pown(G, 2) * lperp)
			    * erf(G * sqrt(lperp - lpar));
}

#endif // DMRICM_ASTROCYLINDERS_CL