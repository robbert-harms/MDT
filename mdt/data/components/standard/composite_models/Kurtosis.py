from mdt import CompositeModelTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Kurtosis(CompositeModelTemplate):
    """The standard Kurtosis model with in vivo defaults."""

    model_expression = '''
        S0 * KurtosisTensor
    '''
    volume_selection = {'unweighted_threshold': 25e6,
                        'use_unweighted': True,
                        'use_weighted': True,
                        'min_bval': 0,
                        'max_bval': 3e9 + 0.1e9}
