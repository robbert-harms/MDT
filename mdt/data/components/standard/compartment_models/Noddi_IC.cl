/**
 * Author = Robbert Harms
 * Date = 2/26/14
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

// do not change this value! It would require adding approximations to the functions below
#define NODDI_IC_MAX_POLYNOMIAL_ORDER 6

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

    MOT_FLOAT_TYPE LePerp = -2 * GAMMA_H_SQ * pown(G, 2) * NeumannCylPerpPGSESum(Delta, delta, d, R);

    MOT_FLOAT_TYPE lgi[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
    Noddi_IC_LegendreGaussianIntegral(LePerp + d * b, lgi);

    MOT_FLOAT_TYPE signal = 0.0;
    for(int i = 0; i < NODDI_IC_MAX_POLYNOMIAL_ORDER + 1; i++){
        signal += lgi[i] * watson_coeff[i] * sqrt((i + 0.25)/M_PI) * getFirstLegendreTerm(cosTheta, 2*i);
    }

    if(signal <= 0){
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
    MOT_FLOAT_TYPE tmp[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];

    if(x > 0.05){
        // exact
        tmp[0] = M_SQRTPI * erf(sqrt(x))/sqrt(x);
        for(int i = 1; i < NODDI_IC_MAX_POLYNOMIAL_ORDER + 1; i++){
            tmp[i] = (-exp(-x) + (i - 0.5) * tmp[i-1]) / x;
        }

        result[0] = tmp[0];
        result[1] = -0.5*tmp[0] + 1.5*tmp[1];
        result[2] = 0.375*tmp[0] - 3.75*tmp[1] + 4.375*tmp[2];
        result[3] = -0.3125*tmp[0] + 6.5625*tmp[1] - 19.6875*tmp[2] + 14.4375*tmp[3];
        result[4] = 0.2734375*tmp[0] - 9.84375*tmp[1] + 54.140625*tmp[2] - 93.84375*tmp[3] + 50.2734375*tmp[4];
        result[5] = -(63/256.0)*tmp[0] + (3465/256.0)*tmp[1] - (30030/256.0)*tmp[2] + (90090/256.0)*tmp[3] - (109395/256.0)*tmp[4] + (46189/256.0)*tmp[5];
        result[6] = (231/1024.0)*tmp[0] - (18018/1024.0)*tmp[1] + (225225/1024.0)*tmp[2] - (1021020/1024.0)*tmp[3] + (2078505/1024.0)*tmp[4] - (1939938/1024.0)*tmp[5] + (676039/1024.0)*tmp[6];
    }
    else{
        // approximate
        tmp[0] = pown(x, 2);
        tmp[1] = tmp[0] * x;
        tmp[2] = tmp[1] * x;
        tmp[3] = tmp[2] * x;
        tmp[4] = tmp[3] * x;

        result[0] = 2 - 2*x/3.0 + tmp[0]/5 - tmp[1]/21.0 + tmp[2]/108.0;
        result[1] = -4*x/15.0 + 4*tmp[0]/35.0 - 2*tmp[1]/63.0 + 2*tmp[2]/297.0;
        result[2] = 8*tmp[0]/315.0 - 8*tmp[1]/693.0 + 4*tmp[2]/1287.0;
        result[3] = -16*tmp[1]/9009.0 + 16*tmp[2]/19305.0;
        result[4] = 32*tmp[2]/328185.0;
        result[5] = -64*tmp[3]/14549535.0;
        result[6] = 128*tmp[4]/760543875.0;
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

    MOT_FLOAT_TYPE tmp[NODDI_IC_MAX_POLYNOMIAL_ORDER];

    if(kappa <= 30){
        MOT_FLOAT_TYPE ks[NODDI_IC_MAX_POLYNOMIAL_ORDER - 1];
        ks[0] = pown(kappa, 2);
        ks[1] = ks[0] * kappa;
        ks[2] = ks[1] * kappa;
        ks[3] = ks[2] * kappa;
        ks[4] = ks[3] * kappa;

        if(kappa > 0.1){
            // exact
            tmp[0] = sqrt(kappa);
            tmp[1] = tmp[0] * kappa;
            tmp[2] = tmp[1] * kappa;
            tmp[3] = tmp[2] * kappa;
            tmp[4] = tmp[3] * kappa;
            tmp[5] = tmp[4] * kappa;

            MOT_FLOAT_TYPE erfik = ferfi(tmp[0]);
            MOT_FLOAT_TYPE ierfik = 1/erfik;
            MOT_FLOAT_TYPE ek = exp(kappa);
            MOT_FLOAT_TYPE dawsonk = M_SQRTPI_2 * erfik/ek;

            result[1] = 3 * tmp[0] - (3 + 2 * kappa) * dawsonk;
            result[1] = sqrt(5.0) * result[1] * ek;
            result[1] = result[1]*ierfik/kappa;

            result[2] = (105 + 60*kappa + 12*ks[0] )*dawsonk;
            result[2] = result[2] -105*tmp[0] + 10*tmp[1];
            result[2] = .375*result[2]*ek/ks[0];
            result[2] = result[2]*ierfik;

            result[3] = -3465 - 1890*kappa - 420*ks[0]  - 40*ks[1] ;
            result[3] = result[3]*dawsonk;
            result[3] = result[3] + 3465*tmp[0] - 420*tmp[1]  + 84*tmp[2];
            result[3] = result[3]*sqrt(13*M_PI)/64/ks[1];
            result[3] = result[3]/dawsonk;

            result[4] = 675675 + 360360*kappa + 83160*ks[0]  + 10080*ks[1]  + 560*ks[2] ;
            result[4] = result[4]*dawsonk;
            result[4] = result[4] - 675675*tmp[0] + 90090*tmp[1]  - 23100*tmp[2]  + 744*tmp[3];
            result[4] = sqrt(17.0)*result[4]*ek;
            result[4] = result[4]/512.0/ks[2];
            result[4] = result[4]*ierfik;

            result[5] = -43648605 - 22972950*kappa - 5405400*ks[0]  - 720720*ks[1]  - 55440*ks[2]  - 2016*ks[3];
            result[5] = result[5]*dawsonk;
            result[5] = result[5] + 43648605*tmp[0] - 6126120*tmp[1]  + 1729728*tmp[2]  - 82368*tmp[3]  + 5104*tmp[4];
            result[5] = sqrt(21*M_PI)*result[5]/4096.0/ks[3];
            result[5] = result[5]/dawsonk;

            result[6] = 7027425405 + 3666482820*kappa + 872972100*ks[0]  + 122522400*ks[1]   + 10810800*ks[2]  + 576576*ks[3]  + 14784*ks[4];
            result[6] = result[6]*dawsonk;
            result[6] = result[6] - 7027425405*tmp[0] + 1018467450*tmp[1]  - 302630328*tmp[2]  + 17153136*tmp[3]  - 1553552*tmp[4]  + 25376*tmp[5];
            result[6] = 5*result[6]*ek;
            result[6] = result[6]/16384.0/ks[4];
            result[6] = result[6]*ierfik;
        }
        else{
            // approximate
            result[1] = (4/3.0*kappa + 8/63.0*ks[0]) * sqrt(M_PI/5.0);
            result[2] = (8/21.0*ks[0] + 32/693.0*ks[1]) * (sqrt(M_PI)*0.2);
            result[3] = (16/693.0*ks[1] + 32/10395.0*ks[2]) * sqrt(M_PI/13);
            result[4] = (32/19305.0*ks[2]) * sqrt(M_PI/17);
            result[5] = 64*sqrt(M_PI/21)*ks[3]/692835.0;
            result[6] = 128*sqrt(M_PI)*ks[4]/152108775.0;
        }
    }
    else{
        // large
        tmp[0] = log(kappa) - log(30.0);
        tmp[1] = tmp[0] * tmp[0];
        tmp[2] = tmp[1] * tmp[0];
        tmp[3] = tmp[2] * tmp[0];
        tmp[4] = tmp[3] * tmp[0];
        tmp[5] = tmp[5] * tmp[0];

        result[1] = 7.52308 + 0.411538*tmp[0] - 0.214588*tmp[1] + 0.0784091*tmp[2] - 0.023981*tmp[3] + 0.00731537*tmp[4] - 0.0026467*tmp[5];
        result[2] = 8.93718 + 1.62147*tmp[0] - 0.733421*tmp[1] + 0.191568*tmp[2] - 0.0202906*tmp[3] - 0.00779095*tmp[4] + 0.00574847*tmp[5];
        result[3] = 8.87905 + 3.35689*tmp[0] - 1.15935*tmp[1] + 0.0673053*tmp[2] + 0.121857*tmp[3] - 0.066642*tmp[4] + 0.0180215*tmp[5];
        result[4] = 7.84352 + 5.03178*tmp[0] - 1.0193*tmp[1] - 0.426362*tmp[2] + 0.328816*tmp[3] - 0.0688176*tmp[4] - 0.0229398*tmp[5];
        result[5] = 6.30113 + 6.09914*tmp[0] - 0.16088*tmp[1] - 1.05578*tmp[2] + 0.338069*tmp[3] + 0.0937157*tmp[4] - 0.106935*tmp[5];
        result[6] = 4.65678 + 6.30069*tmp[0] + 1.13754*tmp[1] - 1.38393*tmp[2] - 0.0134758*tmp[3] + 0.331686*tmp[4] - 0.105954*tmp[5];
    }
}

