from mdt.component_templates.composite_models import DMRICompositeModelTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDI(DMRICompositeModelTemplate):

    description = 'The standard NODDI model'

    model_expression = '''
        S0 * ((Weight(w_csf) * Ball) +
              (Weight(w_ic) * NODDI_IC) +
              (Weight(w_ec) * NODDI_EC)
              )
    '''

    fixes = {'NODDI_IC.d': 1.7e-9,
             'NODDI_IC.R': 0.0,
             'NODDI_EC.d': 1.7e-9,
             'Ball.d': 3.0e-9,
             'NODDI_EC.dperp0': 'NODDI_EC.d * (w_ec.w / (w_ec.w + w_ic.w))',
             'NODDI_EC.kappa': 'NODDI_IC.kappa',
             'NODDI_EC.theta': 'NODDI_IC.theta',
             'NODDI_EC.phi': 'NODDI_IC.phi'}

    post_optimization_modifiers = [
        ('NDI', lambda d: d['w_ic.w'] / (d['w_ic.w'] + d['w_ec.w'])),
        ('ODI', lambda d: d['NODDI_IC.odi'])
    ]

