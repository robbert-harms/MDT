from mdt.components_loader import CompartmentModelsLoader
from mdt.models.single import DMRISingleModelBuilder


__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lc = CompartmentModelsLoader().load


class BallStick(DMRISingleModelBuilder):

    name = 'BallStick'
    ex_vivo_suitable = False
    description = 'The default Ball & Stick model'
    model_expression = '''
        S0 * ( (Weight(Wball) * Ball) +
               (Weight(Wstick) * Stick) )
    '''
    fixes = {'Ball.d': 3.0e-9,
             'Stick.d': 1.7e-9}
    post_optimization_modifiers = [('SNIF', lambda results: 1 - results['Wball.w'])]


class BallStickExVivo(BallStick):

    name = 'BallStick-ExVivo'
    in_vivo_suitable = False
    ex_vivo_suitable = True
    description = 'The Ball & Stick model with ex vivo defaults'
    fixes = {'Ball.d': 2.0e-9,
             'Stick.d': 0.6e-9}


class BallStickStick(DMRISingleModelBuilder):

    name = 'BallStickStick'
    ex_vivo_suitable = False
    description = 'The Ball & 2x Stick model'
    model_listing = (lc('S0'),
                     ((lc('Weight', 'Wball'),
                       lc('Ball').fix('d', 3.0e-9),
                       '*'),
                      ((lc('Weight', 'Wstick0'),
                        lc('Stick', 'Stick0').fix('d', 1.7e-9),
                        '*'),
                       (lc('Weight', 'Wstick1'),
                        lc('Stick', 'Stick1').fix('d', 1.7e-9),
                        '*'),
                       '+'),
                      '+'),
                     '*')
    post_optimization_modifiers = [('SNIF', lambda results: 1 - results['Wball.w'])]


class BallStickStickExVivo(DMRISingleModelBuilder):

    name = 'BallStickStick-ExVivo'
    in_vivo_suitable = False
    ex_vivo_suitable = True
    description = 'The Ball & 2x Stick model with ex vivo defaults'
    model_listing = (lc('S0'),
                     ((lc('Weight', 'Wball'),
                       lc('Ball').fix('d', 2.0e-9),
                       '*'),
                      ((lc('Weight', 'Wstick0'),
                        lc('Stick', 'Stick0').fix('d', 0.6e-9),
                        '*'),
                       (lc('Weight', 'Wstick1'),
                        lc('Stick', 'Stick1').fix('d', 0.6e-9),
                        '*'),
                       '+'),
                      '+'),
                     '*')
    post_optimization_modifiers = [('SNIF', lambda results: 1 - results['Wball.w'])]


class BallStickStickStick(DMRISingleModelBuilder):

    name = 'BallStickStickStick'
    ex_vivo_suitable = False
    description = 'The Ball & 3x Stick model'
    model_listing = (lc('S0'),
                     ((lc('Weight', 'Wball'),
                       lc('Ball').fix('d', 3.0e-9),
                       '*'),
                      ((lc('Weight', 'Wstick0'),
                        lc('Stick', 'Stick0').fix('d', 1.7e-9),
                        '*'),
                       (lc('Weight', 'Wstick1'),
                        lc('Stick', 'Stick1').fix('d', 1.7e-9),
                        '*'),
                       (lc('Weight', 'Wstick2'),
                        lc('Stick', 'Stick2').fix('d', 1.7e-9),
                        '*'),
                       '+'),
                      '+'),
                     '*')
    post_optimization_modifiers = [('SNIF', lambda results: 1 - results['Wball.w'])]


class BallStickStickStickExVivo(DMRISingleModelBuilder):

    name = 'BallStickStickStick-ExVivo'
    ex_vivo_suitable = False
    description = 'The Ball & 3x Stick model with ex vivo defaults'
    model_listing = (lc('S0'),
                     ((lc('Weight', 'Wball'),
                       lc('Ball').fix('d', 2.0e-9),
                       '*'),
                      ((lc('Weight', 'Wstick0'),
                        lc('Stick', 'Stick0').fix('d', 0.6e-9),
                        '*'),
                       (lc('Weight', 'Wstick1'),
                        lc('Stick', 'Stick1').fix('d', 0.6e-9),
                        '*'),
                       (lc('Weight', 'Wstick2'),
                        lc('Stick', 'Stick2').fix('d', 0.6e-9),
                        '*'),
                       '+'),
                      '+'),
                     '*')
    post_optimization_modifiers = [('SNIF', lambda results: 1 - results['Wball.w'])]
