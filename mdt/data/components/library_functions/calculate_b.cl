#ifndef CALCULATE_B_CL
#define CALCULATE_B_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-01
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

double calculate_b(const double G, const double Delta, const double delta){
    return GAMMA_H_SQ * pown(G * delta, 2) * (Delta - (delta/3.0));
}

#endif //CALCULATE_B_CL