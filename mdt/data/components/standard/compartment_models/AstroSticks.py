from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2018-08-28"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AstroSticks(CompartmentTemplate):

    parameters = ('G', 'b', 'd')
    cl_code = '''
        if(b == 0){
            return 1;
        }
        return (sqrt(M_PI) / (2 * G * sqrt((b / (G*G)) * d))) * erf(G * sqrt((b / (G*G)) * d));
    '''
