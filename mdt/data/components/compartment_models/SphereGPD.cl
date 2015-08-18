#ifndef DMRICM_SPHEREGPD_CL
#define DMRICM_SPHEREGPD_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

double cmSphereGPD(const double Delta,
                   const double delta,
                   const double d,
                   const double R,
                   global const double* const CLJnpZeros,
                   const int CLJnpZerosLength){

    double sum = 0;
    double dam;
	double amrdiv;

    // The summation below differs from that of CylinderGPD by having a -2 instead of a -1 in the denominator.
    for(int i = 0; i < CLJnpZerosLength; i++){
        amrdiv = CLJnpZeros[i] / R;
        dam = d * pown(amrdiv, 2);

        sum += (2 * dam * delta
					-  2
					+ (2 * exp(-dam * delta))
					+ (2 * exp(-dam * Delta))
					- exp(-dam * (Delta - delta))
					- exp(-dam * (Delta + delta)))
				/ (pown(dam * amrdiv, 2) * (pown(R * amrdiv, 2) - 2));
    }

    return exp(-2 * GAMMA_H_SQ  * pown(G, 2) * sum);
}

#endif // DMRICM_SPHEREGPD_CL