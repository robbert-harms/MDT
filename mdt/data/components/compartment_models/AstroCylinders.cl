#ifndef DMRICM_ASTROCYLINDERS_CL
#define DMRICM_ASTROCYLINDERS_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

model_float cmAstroCylinders(const double4 g,
                        const double b,
                        const double G,
                        const double Delta,
                        const double delta,
                        const double d,
                        const double R,
                        global const double* const CLJnpZeros,
                        const int CLJnpZerosLength){

    double sum = NeumannCylPerpPGSESum(Delta, delta, d, R, CLJnpZeros, CLJnpZerosLength);

    double lperp = (-2 * GAMMA_H_SQ * sum);
	double lpar = -b * 1.0/pown(G, 2) * d;

    return (sqrt(M_PI) / (2 * G * sqrt(lperp - lpar)))
			    * exp(pown(G, 2) * lperp)
			    * erf(G * sqrt(lperp - lpar));
}

#endif // DMRICM_ASTROCYLINDERS_CL