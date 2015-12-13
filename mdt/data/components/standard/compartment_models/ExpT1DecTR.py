from mdt.models.compartment_models import DMRICompartmentModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT1DecTR(DMRICompartmentModelBuilder):

    config = dict(
        name='T1_IR',
        cl_function_name='cmExpT1DecTR',
        parameter_list=('TR', 'T1'),
        module_name=__name__
    )
