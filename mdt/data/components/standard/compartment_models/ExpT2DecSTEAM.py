from mdt.models.compartments import CompartmentConfig

__author__ = 'Francisco J. Fritz'
__date__ = "2016-09-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ExpT2DecSTEAM(CompartmentConfig):

    parameter_list = ('SEf', 'TE', 'flip_angle', 'Refoc_fa1', 'Refoc_fa2', 'T2')
    cl_code = """
        return pow(0.5, SEf) * sin(flip_angle) * sin(Refoc_fa1) * sin(Refoc_fa2) * exp(-TE / T2);
    """
