from mdt.models.compartment_models import DMRICompartmentModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AstroSticks(DMRICompartmentModelBuilder):

    config = dict(
        name='AstroSticks',
        cl_function_name='cmAstroSticks',
        parameter_list=('g', 'G', 'b', 'd'),
        module_name=__name__
    )
