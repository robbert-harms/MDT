from mdt.components_loader import CompartmentModelsLoader
from mdt.dmri_composite_model import DMRISingleModelBuilder
from mot.parameter_functions.dependencies import SimpleAssignment

__author__ = 'Robbert Harms'
__date__ = "2015-06-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


lc = CompartmentModelsLoader().load

class Noddi(DMRISingleModelBuilder):

    name = 'Noddi'
    ex_vivo_suitable = False
    description = 'The standard Noddi model'

    model_listing = (lc('S0'),
                     ((lc('Weight', 'Wic'),
                       lc('Noddi_IC').fix('d', 1.7e-9).fix('R', 0.0),
                       '*'),
                      (lc('Weight', 'Wec'),
                       lc('Noddi_EC').fix('d', 1.7e-9),
                       '*'),
                      (lc('Weight', 'Wcsf'),
                       lc('Ball').fix('d', 3.0e-9),
                       '*'),
                      '+'),
                     '*')

    dependencies = (
        ('Noddi_EC.dperp0', SimpleAssignment('Noddi_EC.d * (Wec.w / (1 - Wcsf.w + {eps}))'.format(eps=1e-5))),
        ('Noddi_IC.kappa', SimpleAssignment('((1 - Wcsf.w) >= {cutoff}) * Noddi_IC.kappa'.format(cutoff=0.01),
                                            fixed=False)),
        ('Noddi_EC.kappa', SimpleAssignment('Noddi_IC.kappa')),
        ('Noddi_EC.theta', SimpleAssignment('Noddi_IC.theta')),
        ('Noddi_EC.phi', SimpleAssignment('Noddi_IC.phi'))
    )

    post_optimization_modifiers = (
        ('NDI', lambda d: d['Wic.w'] / (d['Wic.w'] + d['Wec.w'])),
        ('SNIF', lambda d: 1 - d['Wcsf.w']),
        ('ODI', lambda d: d['Noddi_IC.odi'])
    )