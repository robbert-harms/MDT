from mdt.component_templates.compartment_models import CompartmentTemplate
import numpy as np


__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDI_IC(CompartmentTemplate):

    parameter_list = ('g', 'b', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'kappa', 'R')
    dependency_list = ('CerfErfi',
                       'MRIConstants',
                       'NeumannCylindricalRestrictedSignal')
    post_optimization_modifiers = [('odi', lambda results: np.arctan2(1.0, results['kappa'] * 10) * 2 / np.pi)]
