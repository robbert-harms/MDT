from mdt.models.compartments import DMRICompartmentModelBuilder

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AstroSticks(DMRICompartmentModelBuilder):

    config = dict(
        name='AstroSticks',
        cl_function_name='cmAstroSticks',
        parameter_list=('g', 'G', 'b', 'd'),
        cl_code_inline='''
            if(b == 0){
                return 1;
            }
            return sqrt(M_PI) / (2 * G * sqrt((b / pown(G, 2)) * d)) * erf(G * sqrt((b /pown(G, 2)) * d));
        '''
    )
