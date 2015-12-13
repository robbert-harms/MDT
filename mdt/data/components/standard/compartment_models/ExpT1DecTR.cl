/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
 * Generate the compartment model signal for the T1 model for TR recovery
 * @params TR the protocol value for TR
 * @params T1 the parameter T1
 */
MOT_FLOAT_TYPE cmExpT1DecTR(const MOT_FLOAT_TYPE TR, const MOT_FLOAT_TYPE T1){
    return abs(1 - exp(-TR / T1));
}

