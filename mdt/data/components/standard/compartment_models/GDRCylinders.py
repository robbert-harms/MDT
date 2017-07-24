from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GDRCylinders(CompartmentTemplate):

    parameter_list = ('g', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'gamma_k', 'gamma_beta', 'gamma_nmr_cyl')
    dependency_list = ('CylinderGPD',)
