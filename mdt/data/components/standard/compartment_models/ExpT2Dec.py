from mdt.models.compartments import DMRICompartmentModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT2Dec(DMRICompartmentModelBuilder):

    config = dict(
        name='T2',
        cl_function_name='cmExpT2Dec',
        parameter_list=('TE', 'T2'),
        cl_code_inline='return exp(-TE / T2);'
    )
