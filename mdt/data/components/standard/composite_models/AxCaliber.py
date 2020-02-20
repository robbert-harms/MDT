from mdt import CompositeModelTemplate

__author__ = 'Robbert Harms'
__date__ = '2017-07-19'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'


class AxCaliber(CompositeModelTemplate):
    """The AxCaliber model with Gamma distributed cylinders."""
    model_expression = '''
        S0 * ((Weight(w_hin) * Tensor) +
              (Weight(w_res) * GDRCylinders))
    '''


class AxCaliber_r1_poisson_tdzep(CompositeModelTemplate):
    """Single fibre population AxCaliber model, Poisson distributed, time-dependent zeppelin.

    The extracellular space modelled by a time-dependent zeppelin tensor.

    This model was originally implemented by Mark Drakesmith, 2018, Cardiff University
    """
    model_expression = '''
        S0 * ((Weight(w_hin0) * TimeDependentZeppelin) +
              (Weight(w_res0) * CylindersPoissonDistr(CylindersPoissonDistr0) ))
    '''

    fixes = {
        'TimeDependentZeppelin.phi': 'CylindersPoissonDistr0.phi',
        'TimeDependentZeppelin.theta': 'CylindersPoissonDistr0.theta',
    }
