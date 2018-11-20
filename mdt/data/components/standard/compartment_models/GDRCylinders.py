from mdt import CompartmentTemplate, FreeParameterTemplate
from mdt.model_building.parameter_functions.transformations import ScaleTransform

__author__ = 'Robbert Harms'
__date__ = '2018-09-15'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class GDRCylinders(CompartmentTemplate):
    """Gamma Distributed Radii cylinders, for use in AxCaliber modelling."""
    parameters = ('g', 'b', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'shape', 'scale', '@cache')
    dependencies = ('VanGelderenCylinder', 'SphericalToCartesian', 'gamma_ppf', 'gamma_pdf')
    cl_code = '''
        uint nmr_radii = 16;
        double radius_spacing = (*cache->upper_radius - *cache->lower_radius) / nmr_radii;

        double direction_2 = pown(dot(g, SphericalToCartesian(theta, phi)), 2);
        double diffusivity_par = -b * d * direction_2;

        double radius;
        double diffusivity_perp;
        double weight_sum = 0;
        double signal_sum = 0;

        for(uint i = 0; i < nmr_radii; i++){
            radius = *cache->lower_radius + (i + 0.5) * radius_spacing;
            
            diffusivity_perp = (1 - direction_2) * VanGelderenCylinder(G, Delta, delta, d, radius);
            signal_sum += cache->weights[i] * exp(diffusivity_par + diffusivity_perp);
            weight_sum += cache->weights[i];
        }
        return signal_sum / weight_sum;
    '''
    cache_info = {
        'fields': ['double lower_radius',
                   'double upper_radius',
                   ('double', 'weights', 16)],
        'cl_code': '''
            *cache->lower_radius = gamma_ppf(0.01, shape, scale);
            *cache->upper_radius = gamma_ppf(0.99, shape, scale);
            
            const uint nmr_radii = 16;
            double radius_spacing = (*cache->upper_radius - *cache->lower_radius) / nmr_radii;
            
            double radius;
            for(uint i = 0; i < nmr_radii; i++){
                radius = *cache->lower_radius + (i + 0.5) * radius_spacing;
                
                // area without * M_PI since it is a constant
                cache->weights[i] = gamma_pdf(radius, shape, scale) * (radius * radius);  
            }
        '''
    }
    extra_optimization_maps = [lambda d: {'R': d['shape'] * d['scale'],
                                          'R_variance': d['shape'] * d['scale'] * d['scale']}]

    class shape(FreeParameterTemplate):
        init_value = 2
        lower_bound = 1e-3
        upper_bound = 25
        sampling_proposal_std = 0.01

    class scale(FreeParameterTemplate):
        init_value = 1e-6
        lower_bound = 0.01e-6
        upper_bound = 20e-6
        parameter_transform = ScaleTransform(1e6)
        sampling_proposal_std = 0.01e-6
