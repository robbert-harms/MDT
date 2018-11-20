from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2018-08-28"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AstroCylinders(CompartmentTemplate):

    parameters = ('b', 'G', 'Delta', 'delta', 'd', 'R')
    dependencies = ['VanGelderenCylinder']
    cl_code = '''
        if(b == 0){
            return 1;
        }
        
        double lperp = VanGelderenCylinder(G, Delta, delta, d, R) / (G * G);
        double lpar = -(b / (G*G)) * d;

        return (sqrt(M_PI) / (2 * G * sqrt(lperp - lpar)))
                    * exp(G * G * lperp)
                    * erf(G * sqrt(lperp - lpar));
    '''
