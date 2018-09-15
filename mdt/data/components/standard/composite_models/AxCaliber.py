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
        S0 * ((Weight(w_ic) * GDRCylinders) +
              (Weight(w_ec) * Zeppelin))
    '''
    fixes = {'Zeppelin.d': 'GDRCylinders.d',
             'Zeppelin.dperp0': 'Zeppelin.d * (w_ec.w / (w_ec.w + w_ic.w))',
             'Zeppelin.theta': 'GDRCylinders.theta',
             'Zeppelin.phi': 'GDRCylinders.phi'}
    extra_optimization_maps = [
        lambda d: {'AxonDensityIndex': (4 * (d['w_ic.w'] / (d['w_ec.w'] + d['w_ic.w'])))
                                        / (np.pi * (2 * d['GDRCylinders.R']) ** 2)}
    ]


class AxCaliber_ExVivo(CompositeModelTemplate):
    """AxCaliber model, for ex-vivo usage. Definition is similar to that of ActiveAx_ExVivo"""
    model_expression = '''
        S0 * ((Weight(w_ic) * GDRCylinders) +
              (Weight(w_ec) * Zeppelin) +
              (Weight(w_csf) * Ball) + 
              (Weight(w_stat) * Dot))
    '''
    inits = {
        'w_csf.w': 0.01,
        'w_stat.w': 0.01
    }
    fixes = {
        'GDRCylinders.d': 0.6e-9,
        'Ball.d': 2.0e-9,
        'Zeppelin.d': 'GDRCylinders.d',
        'Zeppelin.dperp0': 'Zeppelin.d * (w_ec.w / (w_ec.w + w_ic.w))',
        'Zeppelin.theta': 'GDRCylinders.theta',
        'Zeppelin.phi': 'GDRCylinders.phi'
    }
    extra_optimization_maps = [
        lambda d: {'AxonDensityIndex': (4 * (d['w_ic.w'] / (d['w_ec.w'] + d['w_ic.w'])))
                                        / (np.pi * (2 * d['GDRCylinders.R']) ** 2)}
    ]
