#ifndef CALCULATE_B_CL
#define CALCULATE_B_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-01
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

model_float calculate_b(const model_float G, const model_float Delta, const model_float delta){
    return GAMMA_H_SQ * pown(G * delta, 2) * (Delta - (delta/3.0));
}

#endif //CALCULATE_B_CL