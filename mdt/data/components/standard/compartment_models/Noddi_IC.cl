/**
 * Author = Robbert Harms
 * Date = 2/26/14
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

// do not change this value! It would require adding approximations to the functions below
#define NODDI_IC_MAX_POLYNOMIAL_ORDER 6

// sqrt(pi)/2
#ifndef M_SQRTPI_2
#define M_SQRTPI_2 0.8862269254527580
#endif

// 1 / sqrt(pi)
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.5641895835477562
#endif

// sqrt(pi)
#ifndef M_SQRTPI
#define M_SQRTPI 1.7724538509055160
#endif

void Noddi_IC_LegendreGaussianIntegral(const MOT_FLOAT_TYPE x, MOT_FLOAT_TYPE* result);
void Noddi_IC_WatsonSHCoeff(const MOT_FLOAT_TYPE kappa, MOT_FLOAT_TYPE* result);

/**
 * Generate the compartment model signal for the Noddi Intra Cellular (Stick with dispersion) model.
 * If Radius is fixed to 0 the model behaves as a stick (with dispersion), if non-fixed the model behaves as a
 * cylinder (with dispersion).
 *
 * It may seem redundant to have both G/Delta/delta and b as arguments. But that is for speed reasons. b is most
 * of the time available anyway, and G/Delta/delta is only needed if R is not fixed (still, it must be provided for).
 *
 * @params g from the protocol /scheme
 * @params b from the protocol /scheme
 * @params G from the protocol / scheme
 * @params Delta big delta from the protocol / scheme
 * @params delta small delta from the protocol / scheme
 * @params d parameter
 * @params theta parameter
 * @params phi parameter
 * @params kappa parameter (concentration parameter of the Watson's distribution)
 * @params R the radius of the cylinder
 */
MOT_FLOAT_TYPE cmNoddi_IC(const MOT_FLOAT_TYPE4 g,
                          const MOT_FLOAT_TYPE b,
                          const MOT_FLOAT_TYPE G,
                          const MOT_FLOAT_TYPE Delta,
                          const MOT_FLOAT_TYPE delta,
                          const MOT_FLOAT_TYPE d,
                          const MOT_FLOAT_TYPE theta,
                          const MOT_FLOAT_TYPE phi,
                          const MOT_FLOAT_TYPE kappa_non_scaled,
                          const MOT_FLOAT_TYPE R){

    const MOT_FLOAT_TYPE kappa = kappa_non_scaled * 10;

    MOT_FLOAT_TYPE cosTheta = dot(g, (MOT_FLOAT_TYPE4)(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta), 0.0));
    if(fabs(cosTheta) > 1){
        cosTheta = cosTheta / fabs(cosTheta);
    }

    MOT_FLOAT_TYPE watson_coeff[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
    Noddi_IC_WatsonSHCoeff(kappa, watson_coeff);

    MOT_FLOAT_TYPE LePerp = select(-2 * GAMMA_H_SQ * pown(G, 2) * NeumannCylPerpPGSESum(Delta, delta, d, R),
                                   0.0, (long)isequal(R, 0));

    MOT_FLOAT_TYPE lgi[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
    Noddi_IC_LegendreGaussianIntegral(fma(d, b, LePerp), lgi);

    double signal = 0.0;
    for(int i = 0; i < NODDI_IC_MAX_POLYNOMIAL_ORDER + 1; i++){
        signal += lgi[i] * watson_coeff[i] * sqrt((i + 0.25)/M_PI) * getFirstLegendreTerm(cosTheta, 2*i);
    }

    if(signal <= 0 || !isnormal(signal)){
        return 0.00001;
    }

    return exp(LePerp) * signal / 2.0;
}

/**
    Copied from the Matlab Noddi toolbox

    function [L, D] = legendreGaussianIntegral(x, n)
    Computes legendre gaussian integrals up to the order specified and the
    derivatives if requested

    The integral takes the following form, in Mathematica syntax,

    L[x, n] = Integrate[Exp[-x \mu^2] Legendre[2*n, \mu], {\mu, -1, 1}]
    D[x, n] = Integrate[Exp[-x \mu^2] (-\mu^2) Legendre[2*n, \mu], {\mu, -1, 1}]

    original author: Gary Hui Zhang (gary.zhang@ucl.ac.uk)
*/
void Noddi_IC_LegendreGaussianIntegral(const MOT_FLOAT_TYPE x, MOT_FLOAT_TYPE* const result){
    if(x > 0.05){
        // exact
        MOT_FLOAT_TYPE tmp[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];

        tmp[0] = M_SQRTPI * erf(sqrt(x))/sqrt(x);
        for(int i = 1; i < NODDI_IC_MAX_POLYNOMIAL_ORDER + 1; i++){
            tmp[i] = (-exp(-x) + (i - 0.5) * tmp[i-1]) / x;
        }

        result[0] = tmp[0];
        result[1] = fma(tmp[0], (MOT_FLOAT_TYPE)-0.5,         (MOT_FLOAT_TYPE)1.5*tmp[1]);
        result[2] = fma(tmp[0], (MOT_FLOAT_TYPE)0.375,        fma(tmp[1], (MOT_FLOAT_TYPE)-3.75,           (MOT_FLOAT_TYPE)4.375*tmp[2]));
        result[3] = fma(tmp[0], (MOT_FLOAT_TYPE)-0.3125,      fma(tmp[1], (MOT_FLOAT_TYPE)6.5625,          fma(tmp[2], (MOT_FLOAT_TYPE)-19.6875,        (MOT_FLOAT_TYPE)14.4375*tmp[3])));
        result[4] = fma(tmp[0], (MOT_FLOAT_TYPE)0.2734375,    fma(tmp[1], (MOT_FLOAT_TYPE)-9.84375,        fma(tmp[2], (MOT_FLOAT_TYPE)54.140625,       fma(tmp[3], (MOT_FLOAT_TYPE)-93.84375,         (MOT_FLOAT_TYPE)50.2734375*tmp[4]))));
        result[5] = fma(tmp[0], (MOT_FLOAT_TYPE)-(63/256.0),  fma(tmp[1], (MOT_FLOAT_TYPE)(3465/256.0),    fma(tmp[2], (MOT_FLOAT_TYPE)-(30030/256.0),  fma(tmp[3], (MOT_FLOAT_TYPE)(90090/256.0),     fma(tmp[4], (MOT_FLOAT_TYPE)-(109395/256.0),  (MOT_FLOAT_TYPE)(46189/256.0)*tmp[5])))));
        result[6] = fma(tmp[0], (MOT_FLOAT_TYPE)(231/1024.0), fma(tmp[1], (MOT_FLOAT_TYPE)-(18018/1024.0), fma(tmp[2], (MOT_FLOAT_TYPE)(225225/1024.0), fma(tmp[3], (MOT_FLOAT_TYPE)-(1021020/1024.0), fma(tmp[4], (MOT_FLOAT_TYPE)(2078505/1024.0), fma(tmp[5], (MOT_FLOAT_TYPE)-(1939938/1024.0), (MOT_FLOAT_TYPE)(676039/1024.0)*tmp[6]))))));
    }
    else{
        // approximate
        MOT_FLOAT_TYPE x2 = pown(x, 2);
        MOT_FLOAT_TYPE x3 = pown(x, 3);
        MOT_FLOAT_TYPE x4 = pown(x, 4);

        result[0] = 2 - 2*x/3.0 + x2/5 - x3/21.0 + x4/108.0;
        result[1] = -4*x/15.0 + 4*x2/35.0 - 2*x3/63.0 + 2*x4/297.0;
        result[2] = 8*x2/315.0 - 8*x3/693.0 + 4*x4/1287.0;
        result[3] = -16*x3/9009.0 + 16*x4/19305.0;
        result[4] = 32*x4/328185.0;
        result[5] = -64*pown(x, 5)/14549535.0;
        result[6] = 128*pown(x, 6)/760543875.0;
    }
}

/**
    function [C, D] = WatsonSHCoeff(k)
    Computes the spherical harmonic (SH) coefficients of the Watson's
    distribution with the concentration parameter k (kappa) up to the 12th order
    and the derivatives if requested.

    Truncating at the 12th order gives good approximation for kappa up to 64.

    Note that the SH coefficients of the odd orders are always zero.

    author: Gary Hui Zhang (gary.zhang@ucl.ac.uk)
*/
void Noddi_IC_WatsonSHCoeff(const MOT_FLOAT_TYPE kappa, MOT_FLOAT_TYPE* const result){
    result[0] = M_SQRTPI * 2;

    if(kappa > 30){
        // large
        MOT_FLOAT_TYPE lnkd = log(kappa) - log(30.0);
        MOT_FLOAT_TYPE lnkd2 = pown(lnkd, 2);
        MOT_FLOAT_TYPE lnkd3 = pown(lnkd, 3);
        MOT_FLOAT_TYPE lnkd4 = pown(lnkd, 4);
        MOT_FLOAT_TYPE lnkd5 = pown(lnkd, 5);
        MOT_FLOAT_TYPE lnkd6 = pown(lnkd, 6);

        result[1] = fma(lnkd6, (MOT_FLOAT_TYPE)-0.0026467, fma(lnkd5, (MOT_FLOAT_TYPE)0.00731537,  fma(lnkd4, (MOT_FLOAT_TYPE)-0.023981,  fma(lnkd3, (MOT_FLOAT_TYPE)0.0784091, fma(lnkd2, (MOT_FLOAT_TYPE)-0.214588, fma(lnkd, (MOT_FLOAT_TYPE)0.411538, (MOT_FLOAT_TYPE)7.52308))))));
        result[2] = fma(lnkd6, (MOT_FLOAT_TYPE)0.00574847, fma(lnkd5, (MOT_FLOAT_TYPE)-0.00779095, fma(lnkd4, (MOT_FLOAT_TYPE)-0.0202906, fma(lnkd3, (MOT_FLOAT_TYPE)0.191568,  fma(lnkd2, (MOT_FLOAT_TYPE)-0.733421, fma(lnkd, (MOT_FLOAT_TYPE)1.62147, (MOT_FLOAT_TYPE)8.93718))))));
        result[3] = fma(lnkd6, (MOT_FLOAT_TYPE)0.0180215,  fma(lnkd5, (MOT_FLOAT_TYPE)-0.066642,   fma(lnkd4, (MOT_FLOAT_TYPE)0.121857,   fma(lnkd3, (MOT_FLOAT_TYPE)0.0673053, fma(lnkd2, (MOT_FLOAT_TYPE)-1.15935,  fma(lnkd, (MOT_FLOAT_TYPE)3.35689, (MOT_FLOAT_TYPE)8.87905))))));
        result[4] = fma(lnkd6, (MOT_FLOAT_TYPE)-0.0229398, fma(lnkd5, (MOT_FLOAT_TYPE)-0.0688176,  fma(lnkd4, (MOT_FLOAT_TYPE)0.328816,   fma(lnkd3, (MOT_FLOAT_TYPE)-0.426362, fma(lnkd2, (MOT_FLOAT_TYPE)-1.0193,   fma(lnkd, (MOT_FLOAT_TYPE)5.03178, (MOT_FLOAT_TYPE)7.84352))))));
        result[5] = fma(lnkd6, (MOT_FLOAT_TYPE)-0.106935,  fma(lnkd5, (MOT_FLOAT_TYPE)0.0937157,   fma(lnkd4, (MOT_FLOAT_TYPE)0.338069,   fma(lnkd3, (MOT_FLOAT_TYPE)-1.05578,  fma(lnkd2, (MOT_FLOAT_TYPE)-0.16088,  fma(lnkd, (MOT_FLOAT_TYPE)6.09914, (MOT_FLOAT_TYPE)6.30113))))));
        result[6] = fma(lnkd6, (MOT_FLOAT_TYPE)-0.105954,  fma(lnkd5, (MOT_FLOAT_TYPE)0.331686,    fma(lnkd4, (MOT_FLOAT_TYPE)-0.0134758, fma(lnkd3, (MOT_FLOAT_TYPE)-1.38393,  fma(lnkd2, (MOT_FLOAT_TYPE)1.13754,   fma(lnkd, (MOT_FLOAT_TYPE)6.30069, (MOT_FLOAT_TYPE)4.65678))))));

        return;
    }

    MOT_FLOAT_TYPE kappa_2 = pown(kappa, 2);
    MOT_FLOAT_TYPE kappa_3 = pown(kappa, 3);
    MOT_FLOAT_TYPE kappa_4 = pown(kappa, 4);
    MOT_FLOAT_TYPE kappa_5 = pown(kappa, 5);
    MOT_FLOAT_TYPE kappa_6 = pown(kappa, 6);

    if(kappa > 0.1){
        // exact
        MOT_FLOAT_TYPE sqrt_kappa = sqrt(kappa);
        MOT_FLOAT_TYPE sqrt_kappa_2 = sqrt_kappa * kappa;
        MOT_FLOAT_TYPE sqrt_kappa_3 = sqrt_kappa * pown(kappa, 2);
        MOT_FLOAT_TYPE sqrt_kappa_4 = sqrt_kappa * pown(kappa, 3);
        MOT_FLOAT_TYPE sqrt_kappa_5 = sqrt_kappa * pown(kappa, 4);
        MOT_FLOAT_TYPE sqrt_kappa_6 = sqrt_kappa * pown(kappa, 5);

        MOT_FLOAT_TYPE erfik = erfi(sqrt_kappa);
        MOT_FLOAT_TYPE ierfik = 1/erfik;
        MOT_FLOAT_TYPE ek = exp(kappa);
        MOT_FLOAT_TYPE dawsonk = M_SQRTPI_2 * erfik/ek;

        result[1] = (sqrt(5.0) * (3 * sqrt_kappa - (3 + 2 * kappa) * dawsonk) * ek)*ierfik/kappa;
        result[2] = (.375*(((105 + 60*kappa + 12*kappa_2 )*dawsonk) -105*sqrt_kappa + 10*sqrt_kappa_2)*ek/kappa_2)*ierfik;
        result[3] = ((((-3465 - 1890*kappa - 420*kappa_2  - 40*kappa_3 )*dawsonk) + 3465*sqrt_kappa - 420*sqrt_kappa_2  + 84*sqrt_kappa_3)*sqrt(13*M_PI)/64/kappa_3)/dawsonk;

        result[4] = 675675 + 360360*kappa + 83160*kappa_2  + 10080*kappa_3  + 560*kappa_4 ;
        result[4] = result[4]*dawsonk;
        result[4] = result[4] - 675675*sqrt_kappa + 90090*sqrt_kappa_2  - 23100*sqrt_kappa_3  + 744*sqrt_kappa_4;
        result[4] = sqrt(17.0)*result[4]*ek;
        result[4] = result[4]/512.0/kappa_4;
        result[4] = result[4]*ierfik;

        result[5] = -43648605 - 22972950*kappa - 5405400*kappa_2  - 720720*kappa_3  - 55440*kappa_4  - 2016*kappa_5;
        result[5] = result[5]*dawsonk;
        result[5] = result[5] + 43648605*sqrt_kappa - 6126120*sqrt_kappa_2  + 1729728*sqrt_kappa_3  - 82368*sqrt_kappa_4  + 5104*sqrt_kappa_5;
        result[5] = sqrt(21*M_PI)*result[5]/4096.0/kappa_5;
        result[5] = result[5]/dawsonk;

        result[6] = 7027425405 + 3666482820*kappa + 872972100*kappa_2  + 122522400*kappa_3   + 10810800*kappa_4  + 576576*kappa_5  + 14784*kappa_6;
        result[6] = result[6]*dawsonk;
        result[6] = result[6] - 7027425405*sqrt_kappa + 1018467450*sqrt_kappa_2  - 302630328*sqrt_kappa_3  + 17153136*sqrt_kappa_4  - 1553552*sqrt_kappa_5  + 25376*sqrt_kappa_6;
        result[6] = 5*result[6]*ek;
        result[6] = result[6]/16384.0/kappa_6;
        result[6] = result[6]*ierfik;

        return;
    }

    // approximate
    result[1] = (4 / 3.0 * kappa + 8 / 63.0 * kappa_2) * sqrt(M_PI/5.0);
    result[2] = (8 / 21.0 * kappa_2 + 32 / 693.0 * kappa_3) * (M_SQRTPI * 0.2);
    result[3] = (16 / 693.0 * kappa_3 + 32 / 10395.0 * kappa_4) * sqrt(M_PI/13);
    result[4] = (32 / 19305.0 * kappa_4) * sqrt(M_PI/17);
    result[5] = 64 * sqrt(M_PI/21) * kappa_5 / 692835.0;
    result[6] = 128 * M_SQRTPI * kappa_6 / 152108775.0;
}

