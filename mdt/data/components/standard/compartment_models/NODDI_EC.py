from mdt import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NODDI_EC(CompartmentTemplate):

    parameters = ('g', 'b', 'd', 'dperp0', 'theta', 'phi', 'kappa')
    dependencies = ('dawson', 'Zeppelin')
    cl_code = '''
        mot_float_type tmp;
        mot_float_type dw_0, dw_1;
    
        if(kappa > 1e-5){
            tmp = sqrt(kappa)/dawson(sqrt(kappa));
            dw_0 = ( -(d - dperp0) + 2 * dperp0 * kappa + (d - dperp0) * tmp) / (2.0 * kappa);
            dw_1 = ( (d - dperp0) + 2 * (d+dperp0) * kappa - (d - dperp0) * tmp) / (4.0 * kappa);
        }
        else{
            tmp = 2 * (d - dperp0) * kappa;
            dw_0 = ((2 * dperp0 + d) / 3.0) + (tmp/22.5) + ((tmp * kappa) / 236.0);
            dw_1 = ((2 * dperp0 + d) / 3.0) - (tmp/45.0) - ((tmp * kappa) / 472.0);
        }
    
        return Zeppelin(g, b, dw_0, dw_1, theta, phi);
    '''
