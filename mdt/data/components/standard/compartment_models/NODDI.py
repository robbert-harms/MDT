from mdt import CompartmentTemplate, LibraryFunctionTemplate

__author__ = 'Robbert Harms'
__date__ = '2018-09-07'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class NODDI_EC(CompartmentTemplate):
    """The Extra-Cellular compartment of the NODDI-Watson model."""
    parameters = ('g', 'b', 'd', 'dperp0', 'theta', 'phi', 'kappa')
    dependencies = ('dawson', 'Zeppelin')
    cl_code = '''
        mot_float_type tmp;
        mot_float_type dw_0, dw_1;

        if(kappa > 1e-5){
            tmp = sqrt(kappa)/dawson(sqrt(kappa));
            dw_0 = ( -(d - dperp0) + 2 * dperp0 * kappa + (d - dperp0) * tmp) / (2.0 * kappa);
            dw_1 = ( (d - dperp0) + 2 * (d+dperp0) * kappa - (d - dperp0) * tmp) / (4.0 * kappa);
        }
        else{
            tmp = 2 * (d - dperp0) * kappa;
            dw_0 = ((2 * dperp0 + d) / 3.0) + (tmp/22.5) + ((tmp * kappa) / 236.0);
            dw_1 = ((2 * dperp0 + d) / 3.0) - (tmp/45.0) - ((tmp * kappa) / 472.0);
        }

        return Zeppelin(g, b, dw_0, dw_1, theta, phi);
    '''


class NODDI_IC(CompartmentTemplate):
    """Generate the compartment model signal for the NODDI Intra Cellular (Stick with dispersion) compartment.

    This is a transcription from the NODDI matlab toolbox, but with a few changes. Most notably, support for the
    cylindrical model has been removed to simplify the code.
    """
    parameters = ('g', 'b', 'd', 'theta', 'phi', 'kappa')
    dependencies = ('MRIConstants', 'SphericalToCartesian', 'EvenLegendreTerms',
                    'NODDI_IC_LegendreGaussianIntegral', 'NODDI_IC_WatsonSHCoeff')
    cl_code = '''
        mot_float_type LePerp = 0; // used to be "VanGelderenCylinder(G, Delta, delta, d, R)" with R fixed to 0.
        mot_float_type LePar = -d * b;

        mot_float_type watson_sh_coeff[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
        NODDI_IC_WatsonSHCoeff(kappa, watson_sh_coeff);

        mot_float_type lgi[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
        NODDI_IC_LegendreGaussianIntegral(LePerp - LePar, lgi);

        double legendre_terms[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
        EvenLegendreTerms(dot(g, SphericalToCartesian(theta, phi)), 
                          2 * (NODDI_IC_MAX_POLYNOMIAL_ORDER + 1), legendre_terms);

        double signal = 0.0;
        for(int i = 0; i < NODDI_IC_MAX_POLYNOMIAL_ORDER + 1; i++){
            // summing only over the even terms
            signal += watson_sh_coeff[i] * sqrt((i + 0.25)/M_PI_F) * legendre_terms[i] * lgi[i];  
        }

        if(signal < 0){
            signal = 0;
        }

        return exp(LePerp) * signal / 2.0;
    '''

    class NODDI_IC_LegendreGaussianIntegral(LibraryFunctionTemplate):
        """Computes legendre gaussian integrals up to the order specified.

        Copied from the Matlab NODDI toolbox: function [L, D] = legendreGaussianIntegral(x, n)

        The integral takes the following form, in Mathematica syntax,

        L[x, n] = Integrate[Exp[-x \mu^2] Legendre[2*n, \mu], {\mu, -1, 1}]
        D[x, n] = Integrate[Exp[-x \mu^2] (-\mu^2) Legendre[2*n, \mu], {\mu, -1, 1}]

        original author: Gary Hui Zhang (gary.zhang@ucl.ac.uk)
        """
        parameters = [('mot_float_type', 'x'), ('mot_float_type*', 'result')]
        cl_code = '''
            // do not change this value! It would require adding approximations
            #define NODDI_IC_MAX_POLYNOMIAL_ORDER 6
    
            if(x > 0.05){
                // exact
                mot_float_type tmp[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
                tmp[0] = sqrt(M_PI) * erf(sqrt(x))/sqrt(x);
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
                mot_float_type tmp[NODDI_IC_MAX_POLYNOMIAL_ORDER - 1];
                tmp[0] = x * x;
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
        '''

    class NODDI_IC_WatsonSHCoeff(LibraryFunctionTemplate):
        """Computes the spherical harmonic (SH) coefficients of the Watson's distribution up to the 12th order.

        Copied from the Matlab toolbox: function [C, D] = WatsonSHCoeff(k)

        Truncating at the 12th order gives good approximation for kappa up to 64.

        Note that the SH coefficients of the odd orders are always zero.

        original author: Gary Hui Zhang (gary.zhang@ucl.ac.uk)
        """
        parameters = [('mot_float_type', 'kappa'), ('mot_float_type*', 'result')]
        dependencies = ('erfi',)
        cl_code = '''
            // do not change this value! It would require adding approximations
            #define NODDI_IC_MAX_POLYNOMIAL_ORDER 6

            result[0] = sqrt(M_PI) * 2;

            if(kappa <= 30){
                mot_float_type ks[NODDI_IC_MAX_POLYNOMIAL_ORDER - 1];
                ks[0] = kappa * kappa;
                ks[1] = ks[0] * kappa;
                ks[2] = ks[1] * kappa;
                ks[3] = ks[2] * kappa;
                ks[4] = ks[3] * kappa;

                if(kappa > 0.1){
                    // exact
                    mot_float_type sks[NODDI_IC_MAX_POLYNOMIAL_ORDER];
                    sks[0] = sqrt(kappa);
                    sks[1] = sks[0] * kappa;
                    sks[2] = sks[1] * kappa;
                    sks[3] = sks[2] * kappa;
                    sks[4] = sks[3] * kappa;
                    sks[5] = sks[4] * kappa;

                    mot_float_type erfik = erfi(sks[0]);
                    mot_float_type ierfik = 1/erfik;
                    mot_float_type ek = exp(kappa);
                    mot_float_type dawsonk = sqrt(M_PI)/2 * erfik/ek;

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
                    result[3] = result[3]*sqrt(13*M_PI_F)/64/ks[1];
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
                    result[5] = sqrt(21*M_PI_F)*result[5]/4096.0/ks[3];
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
                    result[1] = (4/3.0*kappa + 8/63.0*ks[0]) * sqrt(M_PI_F/5.0);
                    result[2] = (8/21.0*ks[0] + 32/693.0*ks[1]) * (sqrt(M_PI_F)*0.2);
                    result[3] = (16/693.0*ks[1] + 32/10395.0*ks[2]) * sqrt(M_PI_F/13);
                    result[4] = (32/19305.0*ks[2]) * sqrt(M_PI_F/17);
                    result[5] = 64*sqrt(M_PI_F/21)*ks[3]/692835.0;
                    result[6] = 128*sqrt(M_PI_F)*ks[4]/152108775.0;
                }
            }
            else{
                // large
                mot_float_type lnkd[NODDI_IC_MAX_POLYNOMIAL_ORDER];
                lnkd[0] = log(kappa) - log(30.0);
                lnkd[1] = lnkd[0] * lnkd[0];
                lnkd[2] = lnkd[1] * lnkd[0];
                lnkd[3] = lnkd[2] * lnkd[0];
                lnkd[4] = lnkd[3] * lnkd[0];
                lnkd[5] = lnkd[4] * lnkd[0];

                result[1] = 7.52308 + 0.411538*lnkd[0] - 0.214588*lnkd[1] + 0.0784091*lnkd[2] - 0.023981*lnkd[3] + 0.00731537*lnkd[4] - 0.0026467*lnkd[5];
                result[2] = 8.93718 + 1.62147*lnkd[0] - 0.733421*lnkd[1] + 0.191568*lnkd[2] - 0.0202906*lnkd[3] - 0.00779095*lnkd[4] + 0.00574847*lnkd[5];
                result[3] = 8.87905 + 3.35689*lnkd[0] - 1.15935*lnkd[1] + 0.0673053*lnkd[2] + 0.121857*lnkd[3] - 0.066642*lnkd[4] + 0.0180215*lnkd[5];
                result[4] = 7.84352 + 5.03178*lnkd[0] - 1.0193*lnkd[1] - 0.426362*lnkd[2] + 0.328816*lnkd[3] - 0.0688176*lnkd[4] - 0.0229398*lnkd[5];
                result[5] = 6.30113 + 6.09914*lnkd[0] - 0.16088*lnkd[1] - 1.05578*lnkd[2] + 0.338069*lnkd[3] + 0.0937157*lnkd[4] - 0.106935*lnkd[5];
                result[6] = 4.65678 + 6.30069*lnkd[0] + 1.13754*lnkd[1] - 1.38393*lnkd[2] - 0.0134758*lnkd[3] + 0.331686*lnkd[4] - 0.105954*lnkd[5];
            }
        '''


class BinghamNODDI_EN(CompartmentTemplate):
    """The Extra-Neurite tissue model of Bingham NODDI."""
    parameters = ('g', 'b', 'd', 'dperp0', 'theta', 'phi', 'psi', 'k1', 'kw', '@cache')
    dependencies = ['ConfluentHyperGeometricFirstKind', 'SphericalToCartesian', 'Tensor']
    cl_code = '''
        double d_mu_1 = dperp0 + (d - dperp0) * *cache->diff_kappa;
        double d_mu_2 = dperp0 + (d - dperp0) * *cache->diff_beta;
        double d_mu_3 = d + 2*dperp0 - d_mu_1 - d_mu_2;

        return Tensor(g, b, d_mu_1, d_mu_2, d_mu_3, theta, phi, psi);
    '''
    cache_info = {
        'fields': ['double diff_kappa',
                   'double diff_beta'],
        'cl_code': '''
            double kappa = k1;
            double beta = k1 / kw;

            double DELTA = 1e-4;
            double normalization_constant = ConfluentHyperGeometricFirstKind(-kappa, -beta, 0);

            *cache->diff_kappa = (ConfluentHyperGeometricFirstKind(-(kappa+DELTA), -beta, 0) -
                                  ConfluentHyperGeometricFirstKind(-(kappa-DELTA), -beta, 0))
                                  / (2*DELTA) / normalization_constant;

            *cache->diff_beta = (ConfluentHyperGeometricFirstKind(-kappa, -(beta+DELTA), 0) -
                                 ConfluentHyperGeometricFirstKind(-kappa, -(beta-DELTA), 0))
                                 / (2*DELTA) / normalization_constant;
        '''
    }


class BinghamNODDI_IN(CompartmentTemplate):
    """The Intra-Neurite tissue model of Bingham NODDI."""
    parameters = ('g', 'b', 'd', 'theta', 'phi', 'psi', 'k1', 'kw', '@cache')
    dependencies = ['EigenvaluesSymmetric3x3', 'ConfluentHyperGeometricFirstKind', 'TensorSphericalToCartesian']
    cl_code = '''
        double kappa = k1;
        double beta = k1 / kw;
    
        mot_float_type4 v1, v2, v3;
        TensorSphericalToCartesian(theta, phi, psi, &v1, &v2, &v3);

        mot_float_type Q[9];
        Q[0] = pown(dot(g, v3), 2) * (-b * d);
        Q[1] = dot(g, v3) * dot(g, v2) * (-b * d);
        Q[2] = dot(g, v3) * dot(g, v1) * (-b * d);
        Q[3] = Q[1];
        Q[4] = pown(dot(g, v2), 2) * (-b * d);
        Q[5] = dot(g, v2) * dot(g, v1) * (-b * d);
        Q[6] = Q[2];
        Q[7] = Q[5];
        Q[8] = pown(dot(g, v1), 2) * (-b * d);

        Q[4] += beta;
        Q[8] += kappa;

        mot_float_type e[3];
        EigenvaluesSymmetric3x3(Q,e);

        return ConfluentHyperGeometricFirstKind(-e[0], -e[1], -e[2]) / *cache->denom;
    '''
    cache_info = {
        'fields': ['double denom'],
        'cl_code': '''
            *cache->denom = ConfluentHyperGeometricFirstKind(-k1, -(k1 / kw), 0);
        '''
    }
