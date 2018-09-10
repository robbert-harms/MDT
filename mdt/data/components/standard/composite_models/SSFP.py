from mdt import CompositeModelTemplate


class SSFP_BallStick_r1_ExVivo(CompositeModelTemplate):

    name = 'SSFP_BallStick_r1-ExVivo'
    model_expression = '''
        S0 * ( (Weight(w_ball) * SSFP_Ball) +
               (Weight(w_stick0) * SSFP_Stick(SSFP_Stick0)) )
    '''
    fixes = {'SSFP_Ball.d': 2.0e-9,
             'SSFP_Stick0.d': 0.6e-9,
             }
    extra_optimization_maps = [
        lambda d: {'FS': 1 - d['w_ball.w']}
    ]


class SSFP_Tensor_ExVivo(CompositeModelTemplate):

    name = 'SSFP_Tensor-ExVivo'
    model_expression = '''
        S0 * SSFP_Tensor
    '''
    inits = {'SSFP_Tensor.d': 1e-9,
             'SSFP_Tensor.dperp0': 0.6e-10,
             'SSFP_Tensor.dperp1': 0.6e-10}
    volume_selection = None
