from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SphereGPD(CompartmentTemplate):
    parameters = ('G', 'Delta', 'delta', 'd', 'R')
    dependencies = ('VanGelderenSphere',)
    cl_code = '''
        return exp(VanGelderenSphere(G, Delta, delta, d, R));
    '''
