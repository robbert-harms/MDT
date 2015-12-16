from mdt.models.compartments import CompartmentConfig, CLCodeFromInlineString

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Zeppelin(CompartmentConfig):

    name = 'Zeppelin'
    cl_function_name = 'cmZeppelin'
    parameter_list = ('g', 'b', 'd', 'dperp0', 'theta', 'phi')
    cl_code = CLCodeFromInlineString('''
        return exp(-b *
                    (((d - dperp) *
                          pown(dot(g, (MOT_FLOAT_TYPE4)(cos(phi) * sin(theta),
                                                        sin(phi) * sin(theta), cos(theta), 0.0)), 2)
                    ) + dperp));
    ''')
