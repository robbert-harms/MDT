from mdt import CompositeModelTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class S0(CompositeModelTemplate):
    """Models the unweighted signal (aka. b0)."""
    model_expression = 'S0'
    volume_selection = {'b': [(0, 250e6)]}
