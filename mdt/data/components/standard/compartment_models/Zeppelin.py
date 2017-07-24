from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Zeppelin(CompartmentTemplate):

    parameter_list = ('g', 'b', 'd', 'dperp0', 'theta', 'phi')
    cl_code = '''
        mot_float_type4 n = (mot_float_type4)(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta), 0.0);
        return exp(-b * (((d - dperp0) * pown(dot(g, n), 2)) + dperp0));
    '''
