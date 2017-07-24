from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GDRCylinders(CompartmentTemplate):

    parameter_list = ('g', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'gamma_shape', 'gamma_scale',
                      'GDRCylinders_nmr_bins(nmr_bins)')
    dependency_list = ('CylinderGPD', 'GammaFunctions')
