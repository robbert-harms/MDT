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
                    'NODDI_LegendreGaussianIntegral', 'NODDI_WatsonSHCoeff')
    cl_code = '''
        mot_float_type LePerp = 0; // used to be "VanGelderenCylinder(G, Delta, delta, d, R)" with R fixed to 0.
        mot_float_type LePar = -d * b;

        mot_float_type watson_sh_coeff[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
        NODDI_WatsonSHCoeff(kappa, watson_sh_coeff);

        mot_float_type lgi[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
        NODDI_LegendreGaussianIntegral(LePerp - LePar, lgi);

        double legendre_terms[NODDI_IC_MAX_POLYNOMIAL_ORDER + 1];
        EvenLegendreTerms(dot(g, SphericalToCartesian(theta, phi)), NODDI_IC_MAX_POLYNOMIAL_ORDER + 1, legendre_terms);

        double signal = 0.0;
        for(int i = 0; i < NODDI_IC_MAX_POLYNOMIAL_ORDER + 1; i++){
            // summing only over the even terms
            signal += watson_sh_coeff[i] * sqrt((i + 0.25)/M_PI_F) * legendre_terms[i] * lgi[i];  
        }
        
        return exp(LePerp) * fmax(signal, (double)0.0) / 2.0;
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
