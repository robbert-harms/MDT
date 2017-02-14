from mdt.components_config.composite_models import DMRICompositeModelConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class S0(DMRICompositeModelConfig):

    description = 'Models the unweighted signal (aka. b0).'
    model_expression = 'S0'
    volume_selection = {'unweighted_threshold': 250e6,
                        'use_unweighted': True,
                        'use_weighted': False}
