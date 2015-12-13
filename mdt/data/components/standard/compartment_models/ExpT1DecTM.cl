/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

MOT_FLOAT_TYPE cmExpT1DecTM(const MOT_FLOAT_TYPE TM, const MOT_FLOAT_TYPE T1){
    return exp(-TM / T1);
}

