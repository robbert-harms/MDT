from mdt.component_templates.composite_models import DMRICompositeModelTemplate

__author__ = 'Robbert Harms'
__date__ = '2017-07-24'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


# class AxCaliber(DMRICompositeModelTemplate):
#
#     description = '''
#         The AxCaliber model for use in in-vivo measurements.
#     '''
#     model_expression = '''
#         S0 * ((Weight(w_ic) * GDRCylinders) +
#               (Weight(w_ec) * Zeppelin))
#     '''
#     inits = {
#         'w_ec.w': 0.3
#     }
#     fixes = {'Zeppelin.d': 'GDRCylinders.d',
#              'Zeppelin.theta': 'GDRCylinders.theta',
#              'Zeppelin.phi': 'GDRCylinders.phi'}
