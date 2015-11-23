#ifndef DMRICM_SPHEREGPD_CL
#define DMRICM_SPHEREGPD_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

MOT_FLOAT_TYPE cmSphereGPD(const MOT_FLOAT_TYPE Delta,
                        const MOT_FLOAT_TYPE delta,
                        const MOT_FLOAT_TYPE d,
                        const MOT_FLOAT_TYPE R){

    const MOT_FLOAT_TYPE cl_jnp_zeros[] = {
        1.84118378,   5.33144277,   8.53631637,  11.7060049 ,
        14.86358863,  18.01552786,  21.16436986,  24.31132686,
        27.45705057,  30.60192297,  33.7461829 ,  36.88998741,
        40.03344405,  43.17662897,  46.31959756,  49.46239114,
        52.60504111,  55.74757179,  58.8900023 ,  62.03234787
    };
    const int cl_jnp_zeros_length = 20;

    MOT_FLOAT_TYPE sum = 0;
    MOT_FLOAT_TYPE dam;
	MOT_FLOAT_TYPE amrdiv;

    // The summation below differs from that of CylinderGPD by having a -2 instead of a -1 in the denominator.
    for(int i = 0; i < cl_jnp_zeros_length; i++){
        amrdiv = cl_jnp_zeros[i] / R;
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