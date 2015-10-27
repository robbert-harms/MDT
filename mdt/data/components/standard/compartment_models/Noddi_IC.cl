#ifndef DMRICM_NODDIIC_CL
#define DMRICM_NODDIIC_CL

/**
 * Author = Robbert Harms
 * Date = 2/26/14 
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

// do not change this value! It would require adding approximations to the functions below
#define NODDI_IC_MAX_POLYNOMIAL_ORDER 6

void Noddi_IC_LegendreGaussianIntegral(const model_float x, model_float* result);
void Noddi_IC_WatsonSHCoeff(const model_float kappa, model_float* result);
model_float Noddi_IC_CylNeumanLePerp_PGSE(const model_float d, const model_float R, const model_float G,
                                          const model_float Delta,
                                          const model_float delta, global const model_float* const CLJnpZeros,
                                          const int CLJnpZerosLength);

/**
 * See the header for details
 */
model_float cmNoddi_IC(const model_float4 g,
                       const model_float b,
                       const model_float G,
                       const model_float Delta,
                       const model_float delta,
                       const model_float d,
                       const model_float theta,
                       const model_float phi,
                       const model_float kappa_non_scaled,
                       const model_float R,
                       global const model_float* const CLJnpZeros,
                       const int CLJnpZerosLength){

    const model_float kappa = kappa_non_scaled * 10;

    model_float cosTheta = dot(g, (model_float4)(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta), 0.0));
    if(fabs(cosTheta) > 1){
        cosTheta = cosTheta / fabs(cosTheta);
    }

    model_float watson_coeff[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
    Noddi_IC_WatsonSHCoeff(kappa, watson_coeff);

    model_float LePerp = -2 * GAMMA_H_SQ * pown(G, 2) *
                        NeumannCylPerpPGSESum(Delta, delta, d, R, CLJnpZeros, CLJnpZerosLength);
    model_float ePerp = exp(LePerp);
    model_float Lpmp = LePerp + d * b;

    model_float lgi[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
    Noddi_IC_LegendreGaussianIntegral(Lpmp, lgi);

    model_float signal = 0.0;
    for(int i = 0; i < NODDI_IC_MAX_POLYNOMIAL_ORDER + 1; i++){
        signal += lgi[i] * watson_coeff[i] * sqrt((i + 0.25)/M_PI) * getFirstLegendreTerm(cosTheta, 2*i);
    }

    if(signal <= 0){
        return 0.00001;
    }

    return ePerp * signal / 2.0;
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
void Noddi_IC_LegendreGaussianIntegral(const model_float x, model_float* const result){

    if(x > 0.05){
        // exact
        model_float tmp[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
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
        model_float tmp[NODDI_IC_MAX_POLYNOMIAL_ORDER - 1];
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
void Noddi_IC_WatsonSHCoeff(const model_float kappa, model_float* const result){
    result[0] = M_SQRTPI * 2;

    if(kappa <= 30){
        model_float ks[NODDI_IC_MAX_POLYNOMIAL_ORDER - 1];
        ks[0] = pown(kappa, 2);
        ks[1] = ks[0] * kappa;
        ks[2] = ks[1] * kappa;
        ks[3] = ks[2] * kappa;
        ks[4] = ks[3] * kappa;

        if(kappa > 0.1){
            // exact
            model_float sks[NODDI_IC_MAX_POLYNOMIAL_ORDER];
            sks[0] = sqrt(kappa);
            sks[1] = sks[0] * kappa;
            sks[2] = sks[1] * kappa;
            sks[3] = sks[2] * kappa;
            sks[4] = sks[3] * kappa;
            sks[5] = sks[4] * kappa;

            model_float erfik = erfi(sks[0]);
            model_float ierfik = 1/erfik;
            model_float ek = exp(kappa);
            model_float dawsonk = M_SQRTPI_2 * erfik/ek;

            result[1] = 3 * sks[0] - (3 + 2 * kappa) * dawsonk;
            result[1] = sqrt(5.0) * result[1] * ek;
            result[1] = result[1]*ierfik/kappa;

            result[2] = (105 + 60*kappa + 12*ks[0] )*dawsonk;
            result[2] = result[2] -105*sks[0] + 10*sks[1];
            result[2] = .375*result[2]*ek/ks[0];
            result[2] = result[2]*ierfik;

            result[3] = -3465 - 1890*kappa - 420*ks[0]  - 40*ks[1] ;
            result[3] = result[3]*dawsonk;
            result[3] = result[3] + 3465*sks[0] - 420*sks[1]  + 84*sks[2];
            result[3] = result[3]*sqrt(13*M_PI)/64/ks[1];
            result[3] = result[3]/dawsonk;

            result[4] = 675675 + 360360*kappa + 83160*ks[0]  + 10080*ks[1]  + 560*ks[2] ;
            result[4] = result[4]*dawsonk;
            result[4] = result[4] - 675675*sks[0] + 90090*sks[1]  - 23100*sks[2]  + 744*sks[3];
            result[4] = sqrt(17.0)*result[4]*ek;
            result[4] = result[4]/512.0/ks[2];
            result[4] = result[4]*ierfik;

            result[5] = -43648605 - 22972950*kappa - 5405400*ks[0]  - 720720*ks[1]  - 55440*ks[2]  - 2016*ks[3];
            result[5] = result[5]*dawsonk;
            result[5] = result[5] + 43648605*sks[0] - 6126120*sks[1]  + 1729728*sks[2]  - 82368*sks[3]  + 5104*sks[4];
            result[5] = sqrt(21*M_PI)*result[5]/4096.0/ks[3];
            result[5] = result[5]/dawsonk;

            result[6] = 7027425405 + 3666482820*kappa + 872972100*ks[0]  + 122522400*ks[1]   + 10810800*ks[2]  + 576576*ks[3]  + 14784*ks[4];
            result[6] = result[6]*dawsonk;
            result[6] = result[6] - 7027425405*sks[0] + 1018467450*sks[1]  - 302630328*sks[2]  + 17153136*sks[3]  - 1553552*sks[4]  + 25376*sks[5];
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
        model_float lnkd[NODDI_IC_MAX_POLYNOMIAL_ORDER];
        lnkd[0] = log(kappa) - log(30.0);
        lnkd[1] = lnkd[0] * lnkd[0];
        lnkd[2] = lnkd[1] * lnkd[0];
        lnkd[3] = lnkd[2] * lnkd[0];
        lnkd[4] = lnkd[3] * lnkd[0];
        lnkd[5] = lnkd[5] * lnkd[0];

        result[1] = 7.52308 + 0.411538*lnkd[0] - 0.214588*lnkd[1] + 0.0784091*lnkd[2] - 0.023981*lnkd[3] + 0.00731537*lnkd[4] - 0.0026467*lnkd[5];
        result[2] = 8.93718 + 1.62147*lnkd[0] - 0.733421*lnkd[1] + 0.191568*lnkd[2] - 0.0202906*lnkd[3] - 0.00779095*lnkd[4] + 0.00574847*lnkd[5];
        result[3] = 8.87905 + 3.35689*lnkd[0] - 1.15935*lnkd[1] + 0.0673053*lnkd[2] + 0.121857*lnkd[3] - 0.066642*lnkd[4] + 0.0180215*lnkd[5];
        result[4] = 7.84352 + 5.03178*lnkd[0] - 1.0193*lnkd[1] - 0.426362*lnkd[2] + 0.328816*lnkd[3] - 0.0688176*lnkd[4] - 0.0229398*lnkd[5];
        result[5] = 6.30113 + 6.09914*lnkd[0] - 0.16088*lnkd[1] - 1.05578*lnkd[2] + 0.338069*lnkd[3] + 0.0937157*lnkd[4] - 0.106935*lnkd[5];
        result[6] = 4.65678 + 6.30069*lnkd[0] + 1.13754*lnkd[1] - 1.38393*lnkd[2] - 0.0134758*lnkd[3] + 0.331686*lnkd[4] - 0.105954*lnkd[5];
    }
}
#endif // DMRICM_NODDIIC_CL