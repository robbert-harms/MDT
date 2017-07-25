from mdt.component_templates.compartment_models import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GDRCylinders(CompartmentTemplate):

    parameter_list = ('g', 'G', 'Delta', 'delta', 'd', 'theta', 'phi', 'gamma_shape', 'gamma_scale',
                      'GDRCylinders_nmr_bins(nmr_bins)')
    dependency_list = ('CylinderGPD', 'GammaFunctions')
    cl_code = '''
        mot_float_type lower = findGammaCDFCrossing(0, gamma_scale*gamma_shape, 1.0/nmr_bins, 1e-20, 
                                                    gamma_shape, gamma_scale);
        mot_float_type upper = findGammaCDFCrossing(lower, nmr_bins*gamma_scale*gamma_shape, (1-1.0/nmr_bins), 1e-20,
                                                    gamma_shape, gamma_scale);
    
        mot_float_type binWidth = (upper-lower)/nmr_bins;
        mot_float_type weight = 0;
        mot_float_type radius = 0;
        double signal = 0;
    
        for(int bin_index = 0; bin_index < nmr_bins; bin_index++){
            radius = (lower + (bin_index + 0.5) * binWidth);
    
            weight = (gamma_cdf(gamma_shape, gamma_scale, lower + (bin_index + 1)*binWidth)
                         - gamma_cdf(gamma_shape, gamma_scale, lower + bin_index * binWidth)
                      ) / (1 - (2.0/nmr_bins));
    
            signal += weight * CylinderGPD(g, G, Delta, delta, d, theta, phi, radius);
        }
        return signal;
    '''
    cl_extra = '''
        //Using Brent root finding to determine cdfs
        mot_float_type findGammaCDFCrossing(mot_float_type startx, mot_float_type stopx,
                                         const mot_float_type offset, const mot_float_type convergence,
                                         const mot_float_type gamma_shape, const mot_float_type gamma_scale){
        
            int max_iter = 1000;
        
            mot_float_type fstartx = gamma_cdf(gamma_shape, gamma_scale, startx) - offset;
            mot_float_type fstopx = gamma_cdf(gamma_shape, gamma_scale, stopx) - offset;
            mot_float_type delta = fabs(stopx-startx);
        
            if(fstartx * fstopx > 0){
                if (fstartx>0){
                    fstartx = gamma_cdf(gamma_shape, gamma_scale, 0) - offset;
                }
                else if (fstopx<0){
                    fstopx = gamma_cdf(gamma_shape, gamma_scale, stopx/gamma_shape) - offset;
                }
                else{
                    return NAN;
                }
            }
        
            mot_float_type root = startx;
            mot_float_type froot = fstartx;
            bool mflag=1;
            int iter = 0;
            mot_float_type s = 0;
            mot_float_type de = 0;
        
            while(!(delta < convergence || fstartx == 0 || fstopx == 0) && iter < max_iter){
                 if (fstartx != froot && fstopx != froot){
                     //inverse interpolation
                     s = startx * fstopx * froot / ((fstartx-fstopx)*(fstartx-froot));
                     s += stopx * fstartx * froot / ((fstopx-fstartx)*(fstopx-froot));
                     s += root * fstartx * fstopx / ((froot-fstartx)*(froot-stopx));
                 }
                 else{
                     //secant method
                     s = stopx - fstopx * (stopx-startx) / (fstopx-fstartx);
                 }
        
                 //bisection
                 if( !((s >= (3*startx+stopx)/4) && (s<=stopx) || (s<=(3*startx+stopx)/4) && (s>stopx))
                        || mflag && (fabs(s-stopx) >= fabs(stopx-root)/2)
                        || !mflag && (fabs(s-stopx) >= fabs(root-de)/2)
                        || mflag && (fabs(stopx-root) < delta)
                        || !mflag && (fabs(root-delta) < delta)){
                     s = (startx + stopx) / 2;
                     mflag=1;
                 }
                 else{
                     mflag=0;
                 }
        
                 mot_float_type fs=gamma_cdf(gamma_shape, gamma_scale, s) - offset;
        
                 de=root;
                 root=stopx;
                 froot=fstopx;
        
                 if ((fstartx * fs) < 0){
                     stopx=s;
                     fstopx=fs;
                 }
                 else{
                     startx = s;
                     fstartx = fs;
                 }
        
                 if (fabs(fstartx) < fabs(fstopx)){
                     //swap startx and stopx
                     mot_float_type tmp=stopx;
                     mot_float_type ftmp=fstopx;
                     stopx=startx;
                     fstopx=fstartx;
                     startx=tmp;
                     fstartx=ftmp;
                 }
                 delta=fabs(stopx-startx);
                 iter++;
            }
            return s;
        }
    '''
