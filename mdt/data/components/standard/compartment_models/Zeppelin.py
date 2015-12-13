from mdt.models.compartment_models import DMRICompartmentModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Zeppelin(DMRICompartmentModelBuilder):

    config = dict(
        name='Zeppelin',
        cl_function_name='cmZeppelin',
        parameter_list=('g', 'b', 'd', 'dperp0', 'theta', 'phi'),
        module_name=__name__
    )
