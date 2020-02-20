from mdt import CompositeModelTemplate, CompartmentTemplate
import numpy as np

__author__ = 'Robbert Harms'
__date__ = '2019-10-22'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'


class sIVIM(CompositeModelTemplate):
    """Simplified IVIM model with only a diffusion constant and a dirac delta function.

    References:
        [1] Jalnefjord O. et al. Comparison of methods for estimation of the intravoxel incoherent motion (IVIM)
        diffusion coefficient (D) and perfusion fraction (f). Magnetic Resonance Materials in Physics,
        Biology and Medicine (2018) 31:715–723 https://doi.org/10.1007/s10334-018-0697-5
    """
    model_expression = '''
        S0 * ( (Weight(w_perfusion) * sIVIM_PerfusionDeltaFunc) +
               (Weight(w_diffusion) * Ball(diffusion))
               )
    '''
    volume_selection = {'b': [(0, 1e-5), (0.120e9 + 0.01e9, np.inf)]}
    lower_bounds = {'Diffusion.d': 0.5e-9}
    upper_bounds = {'Diffusion.d': 6e-9}

    class sIVIM_PerfusionDeltaFunc(CompartmentTemplate):
        parameters = ('b',)
        cl_code = '''
            return b <= 1e-5;
        '''


class IVIM(CompositeModelTemplate):
    model_expression = '''
        S0 * ( (Weight(w_perfusion) * Ball(Perfusion)) +
               (Weight(w_diffusion) * Ball(Diffusion))
               )
    '''
    lower_bounds = {'Perfusion.d': 6e-9,
                    'Diffusion.d': 0.5e-9}
    upper_bounds = {'Perfusion.d': 200e-9,
                    'Diffusion.d': 6e-9}

