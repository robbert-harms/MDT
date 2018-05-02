from mdt import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Ball(CompartmentTemplate):

    parameters = ('b', 'd')
    cl_code = 'return exp(-d * b);'
