from mdt import CompartmentTemplate, FreeParameterTemplate, LibraryFunctionTemplate

__author__ = 'Robbert Harms'
__date__ = '2019-11-14'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'


class CylindersPoissonDistr(CompartmentTemplate):
    """Poisson Distributed Radii cylinders, for use in AxCaliber modelling."""
    description = '''
        Compartment model for a distribution of VanGelderen Cylinders with a continuous Poisson distribution
    '''
    parameters = ('g', 'b', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'poisson_lambda', '@cache')
    dependencies = ('VanGelderenCylinder', 'SphericalToCartesian', 'PoissonPDF',)
    cl_code = '''
        double direction_2 = pown(dot(g, SphericalToCartesian(theta, phi)), 2);
        double diffusivity_par = -b * d * direction_2;

        double radius;
        double diffusivity_perp;
        double weight_sum = 0;
        double signal_sum = 0;

        for(uint i = 0; i < PDRCylinders_nmr_radii; i++){
            radius = PDRCylinders_radii[i] * 1e-6;

            diffusivity_perp = (1 - direction_2) * VanGelderenCylinder(G, Delta, delta, d, radius);
            signal_sum += cache->weights[i] * exp(diffusivity_par + diffusivity_perp);
            weight_sum += cache->weights[i];
        }
        return signal_sum / weight_sum;
    '''
    cl_extra = '''
        constant const int PDRCylinders_nmr_radii = 18;
        constant const double PDRCylinders_radii[] = {
            0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7.5, 10, 15
        }; // micrometers
    '''
    cache_info = {
        'fields': [('double', 'weights', 18)],
        'cl_code': '''
            double radius;
            for(uint i = 0; i < PDRCylinders_nmr_radii; i++){
                radius = PDRCylinders_radii[i];

                // area without * M_PI since it is a constant
                cache->weights[i] = PoissonPDF(poisson_lambda, radius)  * (radius * radius);
            }
        '''
    }
    extra_optimization_maps = [
        lambda results: {'diameter': results['poisson_lambda'] * 2}]

    class poisson_lambda(FreeParameterTemplate):
        init_value = 1
        lower_bound = 1e-5
        upper_bound = 20
        parameter_transform = 'CosSqrClamp'
        sampling_proposal_std = 0.01

    class PoissonPDF(LibraryFunctionTemplate):
        description = '''
            Computes the (continuous) Poisson probability density function, parameterised by lambda.

            This computes the Poisson PDF as: :math:`{\frac {\lamdbda^{x} e^{-\lambda}}{\Gamma (x+1)}`

            With x the desired position, :math:`\lambda` the paramater.
        '''
        return_type = 'double'
        parameters = ['double lambda',
                      'double x']
        cl_code = '''
            return ( pow(lambda, x) * exp(-lambda)) / (tgamma(x+1));
        '''
