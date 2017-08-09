from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Stick(CompartmentTemplate):

    parameter_list = ('g', 'b', 'd', 'theta', 'phi')
    dependency_list = ['SphericalToCartesian']
    cl_code = '''
        return exp(-b * d * pown(dot(g, SphericalToCartesian(theta, phi)), 2));
    '''
