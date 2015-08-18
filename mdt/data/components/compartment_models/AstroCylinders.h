#ifndef DMRICM_ASTROCYLINDERS_H
#define DMRICM_ASTROCYLINDERS_H

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

double cmAstroCylinders(const double4 g,
                        const double b,
                        const double G,
                        const double Delta,
                        const double delta,
                        const double d,
                        const double R,
                        global const double* const CLJnpZeros,
                        const int CLJnpZerosLength);

#endif // DMRICM_ASTROCYLINDERS_H