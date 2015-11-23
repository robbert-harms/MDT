#ifndef NEUMANN_CYL_PERP_PGSE_SUM_CL
#define NEUMANN_CYL_PERP_PGSE_SUM_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-01
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * See the header definition for explanation
 */
MOT_FLOAT_TYPE NeumannCylPerpPGSESum(const MOT_FLOAT_TYPE Delta,
                                  const MOT_FLOAT_TYPE delta,
                                  const MOT_FLOAT_TYPE d,
                                  const MOT_FLOAT_TYPE R,
                                  global const MOT_FLOAT_TYPE* const CLJnpZeros,
                                  const int CLJnpZerosLength){
    if(R == 0.0){
        return 0;
    }

    MOT_FLOAT_TYPE sum = 0;
    MOT_FLOAT_TYPE dam;
	MOT_FLOAT_TYPE amrdiv;

    for(int i = 0; i < CLJnpZerosLength; i++){
        amrdiv = CLJnpZeros[i] / R;
        dam = d * pown(amrdiv, 2);

        sum += (2 * dam * delta
					-  2
					+ (2 * exp(-dam * delta))
					+ (2 * exp(-dam * Delta))
					- exp(-dam * (Delta - delta))
					- exp(-dam * (Delta + delta)))
				/ (pown(dam * amrdiv, 2) * (pown(R * amrdiv, 2) - 1));
    }
    return sum;
}

#endif //NEUMANN_CYL_PERP_PGSE_SUM_CL