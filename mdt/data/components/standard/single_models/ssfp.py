from mdt.models.single import DMRISingleModelConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SSFP_BallStick(DMRISingleModelConfig):

    name = 'SSFP_BallStick'
    ex_vivo_suitable = False
    description = 'The SSFP Ball & Stick model'
    model_expression = '''
        S0 * ( (Weight(w_ball) * Ball) +
               (Weight(w_stick) * SSFP_Stick) )
    '''
    fixes = {'Ball.d': 3.0e-9,
             'SSFP_Stick.d': 1.7e-9}
    post_optimization_modifiers = [('FS', lambda results: 1 - results['w_ball.w'])]
