import numpy as np
from mdt import CompositeModelTemplate

__author__ = 'Robbert Harms'
__date__ = '2017-07-19'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class AxCaliber(CompositeModelTemplate):
    """The AxCaliber model with Gamma distributed cylinders."""
    model_expression = '''
        S0 * ((Weight(w_hin) * Tensor) +
              (Weight(w_res) * GDRCylinders))
    '''
