import numpy as np
from mdt.components_config.composite_models import DMRICompositeModelConfig
from mdt.components_loader import bind_function, component_import
from mot.model_building.evaluation_models import GaussianEvaluationModel


class SSFP_BallStick_r1_ExVivo(DMRICompositeModelConfig):

    name = 'SSFP_BallStick_r1-ExVivo'
    ex_vivo_suitable = True
    in_vivo_suitable = False
    description = 'The SSFP Ball & Stick model'
    model_expression = '''
        S0 * ( (Weight(w_ball) * SSFP_Ball) +
               (Weight(w_stick) * SSFP_Stick) )
    '''
    # fixes = {'SSFP_Ball.d': 2.0e-9,
    #          'SSFP_Stick.d': 0.6e-9}
    dependencies = {'SSFP_Stick.d': 'SSFP_Ball.d',
                    'SSFP_Stick.T1_exvivo': 'SSFP_Ball.T1_exvivo',
                    'SSFP_Stick.T2_exvivo': 'SSFP_Ball.T2_exvivo'
                    }
    post_optimization_modifiers = [('FS', lambda results: 1 - results['w_ball.w'])]


class SSFP_Tensor_ExVivo(component_import('standard.composite_models.Tensor', 'TensorExVivo')):

    name = 'SSFP_Tensor-ExVivo'
    description = 'The SSFP Tensor model with ex vivo defaults.'
    model_expression = '''
        S0 * SSFP_Tensor
    '''
    inits = {'Tensor.d': 1e-9,
             'Tensor.dperp0': 0.6e-10,
             'Tensor.dperp1': 0.6e-10}
    volume_selection = None
