from mdt import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CHARMEDRestricted(CompartmentTemplate):

    parameters = ('g', 'b', 'G', 'Delta', 'delta', 'TE', 'd', 'theta', 'phi')
    dependencies = ('SphericalToCartesian', 'NeumanCylinderLongApprox')
    cl_code = '''
        const mot_float_type direction_2 = pown(dot(g, SphericalToCartesian(theta, phi)), 2);
        const mot_float_type signal_par = -b * d * direction_2;
        
        float weights[] = {0.021184720085574, 0.107169623942214, 0.194400551313197,
                           0.266676876170322, 0.214921653661151, 0.195646574827541};
        float radii[] = {1.5e-6, 2.5e-6, 3.5e-6, 4.5e-6, 5.5e-6, 6.5e-6}; // meters
        
        double sum = 0;
        mot_float_type signal_perp;
        
        #pragma unroll
        for(uint i = 0; i < 6; i++){
            signal_perp = (1 - direction_2) * (delta * delta) * NeumanCylinderLongApprox(G, TE/2.0, d, radii[i]);
            sum += weights[i] * exp(signal_par + signal_perp);
        }
        
        return sum;
    '''
