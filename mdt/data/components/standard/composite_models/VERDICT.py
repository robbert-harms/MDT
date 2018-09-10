from mdt import CompositeModelTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class VERDICT(CompositeModelTemplate):
    """Implements the VERDICT colorectal model, as in Panagiotaki et al Cancer Research 2014 paper.
        
    The diffusivity parameter is assumed the same in the intracellular (SphereGPD) amd extracellular (Ball)
    compartments dR = d|| = 9e-10 m^2/s.

    The weights, ``w_vasc``, ``w_ees`` and ``w_ic`` stand for weight vascular, extracellular-extravascular space
    and intracellular.
    """
    model_expression = '''
        S0 * ( (Weight(w_vasc) * Stick) +
               (Weight(w_ees) * Ball) +
               (Weight(w_ic) * SphereGPD))
               )
    '''
    fixes = {'Ball.d': 0.9e-9,
             'SphereGPD.d': 0.9e-9}
