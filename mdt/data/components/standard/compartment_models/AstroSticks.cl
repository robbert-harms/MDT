#ifndef DMRICM_ASTROSTICKS_CL
#define DMRICM_ASTROSTICKS_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

MOT_FLOAT_TYPE cmAstroSticks(const MOT_FLOAT_TYPE4 g,
                          const MOT_FLOAT_TYPE G,
                          const MOT_FLOAT_TYPE b,
                          const MOT_FLOAT_TYPE d){
    if(b == 0){
        return 1;
    }
    return sqrt(M_PI) / (2 * G * sqrt((b / pown(G, 2)) * d))
                * erf(G * sqrt((b /pown(G, 2)) * d));
}

#endif // DMRICM_ASTROSTICKS_CL