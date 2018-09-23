from mdt import CompartmentTemplate


__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Stick_dot(CompartmentTemplate):

    parameters = ('g', 'theta', 'phi', '@cache')
    dependencies = ['SphericalToCartesian']
    cl_code = '''
        return dot(g, *cache->n);
    '''
    cache_info = {
        'fields': [('mot_float_type4', 'n')],
        'cl_code': '''
            *cache->n = SphericalToCartesian(theta, phi);
        '''
    }


class Sticker(CompartmentTemplate):

    parameters = ('g', 'b', 'd', 'theta', 'phi', '@cache')
    dependencies = ['SphericalToCartesian', 'Stick_dot']
    cl_code = '''
        printf("%f", (*cache->dummy));
        return exp(-b * d * pown(Stick_dot(g, theta, phi, *cache->Stick_dot), 2));
    '''
    cache_info = {
        'fields': [('float', 'dummy'),
                   ('float', 'd2', 5)],
        'cl_code': '''
            *cache->dummy = 1;
        '''
    }


class Stick(CompartmentTemplate):

    parameters = ('g', 'b', 'd', 'theta', 'phi')
    dependencies = ['SphericalToCartesian']
    cl_code = '''
        return exp(-b * d * pown(dot(g, SphericalToCartesian(theta, phi)), 2));
    '''
