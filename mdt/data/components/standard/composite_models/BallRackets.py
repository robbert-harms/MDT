from collections import OrderedDict

from mdt import CompositeModelTemplate
import numpy as np

from mdt.lib.post_processing import get_sort_modifier

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BallRacket_r1(CompositeModelTemplate):

    model_expression = '''
        S0 * ( (Weight(w_ball) * Ball) +
               (Weight(w_res0) * Racket(Racket0)) )
    '''
    fixes = {'Racket0.d': 'Ball.d'}

    extra_optimization_maps = [
        lambda results: {'FS': results['w_res0.w']},
        lambda results: {'FS.std': results['w_res0.w.std']}
    ]
    extra_sampling_maps = [
        lambda samples: {'FS': np.mean(samples['w_res0.w'], axis=1),
                         'FS.std': np.std(samples['w_res0.w'], axis=1)}
    ]


class BallRacket_r2(CompositeModelTemplate):

    model_expression = '''
        S0 * ( (Weight(w_ball) * Ball) +
               (Weight(w_res0) * Racket(Racket0)) +
               (Weight(w_res1) * Racket(Racket1)) )
    '''
    fixes = {'Racket0.d': 'Ball.d', 
             'Racket1.d': 'Ball.d'}

    get_sort_modifier(OrderedDict([
        ('w_res0.w', ('w_res0', 'Racket0')),
        ('w_res1.w', ('w_res1', 'Racket1'))
    ]))

    extra_optimization_maps = [
        lambda results: {'FS': 1 - results['w_ball.w']},
        lambda results: {'FS.std': results['w_ball.w.std']}
    ]
    extra_sampling_maps = [
        lambda samples: {
            'FS': np.mean(samples['w_res0.w'] + samples['w_res1.w'], axis=1),
            'FS.std': np.std(samples['w_res0.w'] + samples['w_res1.w'], axis=1)}
    ]

    extra_prior = 'return w_res1.w < w_res0.w;'
