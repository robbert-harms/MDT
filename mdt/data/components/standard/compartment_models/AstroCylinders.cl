#ifndef DMRICM_ASTROCYLINDERS_CL
#define DMRICM_ASTROCYLINDERS_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

MOT_FLOAT_TYPE cmAstroCylinders(const MOT_FLOAT_TYPE4 g,
                             const MOT_FLOAT_TYPE b,
                             const MOT_FLOAT_TYPE G,
                             const MOT_FLOAT_TYPE Delta,
                             const MOT_FLOAT_TYPE delta,
                             const MOT_FLOAT_TYPE d,
                             const MOT_FLOAT_TYPE R,
                             global const MOT_FLOAT_TYPE* const CLJnpZeros,
                             const int CLJnpZerosLength){

    MOT_FLOAT_TYPE sum = NeumannCylPerpPGSESum(Delta, delta, d, R, CLJnpZeros, CLJnpZerosLength);

    MOT_FLOAT_TYPE lperp = (-2 * GAMMA_H_SQ * sum);
	MOT_FLOAT_TYPE lpar = -b * 1.0/pown(G, 2) * d;

    return (sqrt(M_PI) / (2 * G * sqrt(lperp - lpar)))
			    * exp(pown(G, 2) * lperp)
			    * erf(G * sqrt(lperp - lpar));
}

#endif // DMRICM_ASTROCYLINDERS_CL