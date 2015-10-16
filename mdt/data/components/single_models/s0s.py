from mdt.components_loader import CompartmentModelsLoader
from mdt.dmri_composite_model import DMRISingleModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lc = CompartmentModelsLoader().load


class S0(DMRISingleModelBuilder):

    name = 's0'
    description = 'Models the unweighted signal (aka. b0).'
    model_listing = (lc('S0'),)


class S0T2(DMRISingleModelBuilder):

    name = 's0-T2'
    description = 'Models the unweighted signal (aka. b0) with an extra T2.'
    model_listing = (lc('S0'),
                     lc('ExpT2Dec'),
                     '*')


class S0T2T2(DMRISingleModelBuilder):

    name = 's0-T2T2'
    description = 'Model for the unweighted signal with two T2 models, one for short T2 and one for long T2.'
    model_listing = (lc('S0'),
                     ((lc('Weight', 'Wlong'),
                       lc('ExpT2Dec', 'T2long').fix('T2', 0.5),
                       '*'),
                      (lc('Weight', 'Wshort'),
                       lc('ExpT2Dec', 'T2short').ub('T2', 0.08),
                       '*'),
                      '+'),
                     '*')

    post_optimization_modifiers = (
        ('T2short.T2Weighted', lambda d: d['Wshort.w'] * d['T2short.T2']),
        ('T2long.T2Weighted', lambda d: d['Wlong.w'] * d['T2long.T2']),
        ('T2.T2', lambda d: d['T2short.T2Weighted'] + d['T2long.T2Weighted'])
    )
