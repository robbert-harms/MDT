from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDI_EC(CompartmentTemplate):

    parameter_list = ('g', 'b', 'd', 'dperp0', 'theta', 'phi', 'kappa')
    dependency_list = ('CerfDawson', 'Zeppelin')
    cl_code = '''
        const mot_float_type kappa_scaled = kappa * 10;
        mot_float_type tmp;
        mot_float_type dw_0, dw_1;
    
        if(kappa_scaled > 1e-5){
            tmp = sqrt(kappa_scaled)/dawson(sqrt(kappa_scaled));
            dw_0 = ( -(d - dperp0) + 2 * dperp0 * kappa_scaled + (d - dperp0) * tmp) / (2.0 * kappa_scaled);
            dw_1 = ( (d - dperp0) + 2 * (d+dperp0) * kappa_scaled - (d - dperp0) * tmp) / (4.0 * kappa_scaled);
        }
        else{
            tmp = 2 * (d - dperp0) * kappa_scaled;
            dw_0 = ((2 * dperp0 + d) / 3.0) + (tmp/22.5) + ((tmp * kappa_scaled) / 236.0);
            dw_1 = ((2 * dperp0 + d) / 3.0) - (tmp/45.0) - ((tmp * kappa_scaled) / 472.0);
        }
    
        return Zeppelin(g, b, dw_0, dw_1, theta, phi);
    '''
