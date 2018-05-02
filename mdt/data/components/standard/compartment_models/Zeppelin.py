from mdt import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Zeppelin(CompartmentTemplate):

    parameters = ('g', 'b', 'd', 'dperp0', 'theta', 'phi')
    dependencies = ['SphericalToCartesian']
    cl_code = '''
        return exp(-b * (((d - dperp0) * pown(dot(g, SphericalToCartesian(theta, phi)), 2)) + dperp0));
    '''
