#ifndef DMRICM_ASTROSTICKS_CL
#define DMRICM_ASTROSTICKS_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

double cmAstroSticks(const double4 g,
                     const double G,
                     const double b,
                     const double d){
    if(b == 0){
        return 1;
    }
    return sqrt(M_PI) / (2 * G * sqrt((b / pown(G, 2)) * d))
                * erf(G * sqrt((b /pown(G, 2)) * d));
}

#endif // DMRICM_ASTROSTICKS_CL