import numpy as np
from mdt import CompartmentTemplate, CompositeModelTemplate, FreeParameterTemplate, ProtocolParameterTemplate, \
    LibraryFunctionTemplate, CascadeTemplate
from mdt.model_building.parameter_functions.transformations import ScaleTransform
from mdt.utils import voxelwise_vector_matrix_vector_product, create_covariance_matrix
from mot.library_functions import simpsons_rule


class LineshapeGaussian(LibraryFunctionTemplate):
    return_type = 'double'
    parameters = ['double f', 'double T2_b']
    cl_code = '''
        return sqrt(1.0/(2.0*M_PI)) * T2_b * exp(-pown(2.0 * M_PI * f * T2_b, 2) / 2.0);
    '''


class LineshapeLorentzian(LibraryFunctionTemplate):
    return_type = 'double'
    parameters = ['double f', 'double T2_b']
    cl_code = '''
        return (T2_b / M_PI) / (1.0 + pown(2.0 * M_PI * f * T2_b, 2));
    '''


class LineshapeSuperLorentzian(LibraryFunctionTemplate):
    return_type = 'double'
    parameters = ['double f', 'double T2_b']
    dependencies = ['superlorentzian_integrand', simpsons_rule('superlorentzian_integrand')]
    cl_code = '''
        superlorentzian_integrand_data data = {f, T2_b};
        return simpsons_rule_superlorentzian_integrand(0, 1, 50, &data);
    '''

    class superlorentzian_integrand(LibraryFunctionTemplate):
        return_type = 'double'
        parameters = ['double x', 'void* data']
        cl_code = '''
            double T2_b = ((superlorentzian_integrand_data*)data)->T2_b;
            double f = ((superlorentzian_integrand_data*)data)->f;

            double uterm = fabs(T2_b / (3.0 * x * x - 1.0));
            return sqrt(2.0/M_PI) * uterm * exp(-2.0 * pown(2.0 * M_PI * f * uterm, 2));
        '''
        cl_extra = '''
            typedef struct superlorentzian_integrand_data{
                double f; 
                double T2_b;
            } superlorentzian_integrand_data;
        '''


class LineshapeSuperLorentzianInterpolated(LibraryFunctionTemplate):
    """Interpolated super Lorentzian lineshape

    The required data for this lineshape was computed using the Python code::

        T2_b = 1e-5
        freq_count = 150
        freq_start = 500
        freq_stepsize = 500

        frequenties = np.linspace(freq_start, freq_start + freq_stepsize * (freq_count - 1), freq_count)

        grid = SuperLorentzian()().evaluate((frequenties, T2_b), freq_count)

    This can be used instead of the SuperLorentzian lineshape. If you need different interpolation steps, overwrite
    this template with a different interpolation matrix.
    """
    return_type = 'double'
    parameters = ['double f', 'double T2_b']
    dependencies = ['linear_cubic_interpolation']
    cl_code = '''
        double T2_b_scale = 1e-5;
        uint freq_count = 150;
        uint freq_start = 500;
        uint freq_stepsize = 500;

        double scale = T2_b / T2_b_scale;
        double sf = (f * scale - freq_start) / freq_stepsize;

        return scale * linear_cubic_interpolation(sf, freq_count, LineshapeSuperLorentzianInterpolated_lookup_table);
    '''
    cl_extra = '''
        constant float LineshapeSuperLorentzianInterpolated_lookup_table[] = {
             1.6351068497997384e-05, 1.3115492795811157e-05, 1.1241666681196103e-05, 9.869994321088873e-06, 
             8.79781041979886e-06, 7.914196403750282e-06, 7.159953545164688e-06, 6.501822309129753e-06, 
             5.9187370092896205e-06, 5.396337299169265e-06, 4.9244567517431185e-06, 4.495680550607755e-06, 
             4.104430808410283e-06, 3.7463832161993373e-06, 3.4180920723477155e-06, 3.1167423473158515e-06, 
             2.8399801093075133e-06, 2.5857933549322158e-06, 2.352426901287217e-06, 2.138321167528001e-06, 
             1.9420680293872054e-06, 1.7623789684262465e-06, 1.5980621239498786e-06, 1.448005833752447e-06, 
             1.3111669297816838e-06, 1.1865625196512177e-06, 1.0732643058325794e-06, 9.70394723648105e-07, 
             8.771243496989519e-07, 7.926701628937718e-07, 7.16294341733855e-07, 6.473033610647276e-07, 
             5.850472142475361e-07, 5.289186364606442e-07, 4.783522444252163e-07, 4.3282353928886254e-07, 
             3.918477441733863e-07, 3.549784671059634e-07, 3.2180619458758904e-07, 2.919566316414404e-07, 
             2.6508891145979857e-07, 2.4089370228913683e-07, 2.1909124145582068e-07, 1.9942932688388915e-07, 
             1.816812954884552e-07, 1.6564401579574187e-07, 1.5113591935325928e-07, 1.3799509221911866e-07, 
             1.2607744428542744e-07, 1.1525497058546612e-07, 1.0541411520910552e-07, 9.645424512226031e-08, 
             8.828623813911807e-08, 8.083118658736193e-08, 7.401921586922215e-08, 6.77884151670253e-08, 
             6.208387596636248e-08, 5.6856832855435643e-08, 5.2063900178291605e-08, 4.76663975386842e-08, 
             4.3629756832484325e-08, 3.992300339053272e-08, 3.651830390146855e-08, 3.3390574017895954e-08, 
             3.051713889451962e-08, 2.7877440332283366e-08, 2.545278468063438e-08, 2.322612615740714e-08, 
             2.1181880763075917e-08, 1.930576647769909e-08, 1.7584665922739697e-08, 1.6006508137241576e-08, 
             1.456016655247463e-08, 1.3235370647395598e-08, 1.2022629127292448e-08, 1.0913162789529515e-08, 
             9.898845524315148e-09, 8.972152146678366e-09, 8.126111970774798e-09, 7.35426722203145e-09, 
             6.650635539463799e-09, 6.009675952783081e-09, 5.426257829619494e-09, 4.895632380138091e-09, 
             4.413406382140566e-09, 3.975517851821481e-09, 3.5782134358287956e-09, 3.218027341065111e-09, 
             2.8917616514154244e-09, 2.5964679067291395e-09, 2.329429840138001e-09, 2.0881471861869747e-09, 
             1.8703204851526994e-09, 1.673836819037118e-09, 1.4967564226374533e-09, 1.3373001192848592e-09, 
             1.1938375356972415e-09, 1.0648760542152894e-09, 9.490504637290755e-10, 8.451132730484687e-10, 
             7.519256524742328e-10, 6.684489710045162e-10, 5.937368980523485e-10, 5.269280398213664e-10, 
             4.672390816396365e-10, 4.139584086223556e-10, 3.664401780502923e-10, 3.240988178310485e-10, 
             2.8640392636758843e-10, 2.5287555010143854e-10, 2.2307981593159215e-10, 1.9662489663708468e-10, 
             1.7315728835226004e-10, 1.5235838005851848e-10, 1.339412959635643e-10, 1.1764799253708605e-10, 
             1.0324659285825612e-10, 9.052894180324604e-11, 7.930836645783098e-11, 6.94176269789315e-11, 
             6.070704394759365e-11, 5.304278905255795e-11, 4.6305326716595796e-11, 4.038799502575617e-11, 
             3.51957150433901e-11, 3.064381828534206e-11, 2.6656982799280144e-11, 2.316826892927348e-11, 
             2.0118246456090952e-11, 1.7454205384183292e-11, 1.5129443197966466e-11, 1.3102621933071603e-11, 
             1.133718890302249e-11, 9.800855388841954e-12, 8.465128038970233e-12, 7.304888140311299e-12, 
             6.298014308985523e-12, 5.425044512317307e-12, 4.668873672633755e-12, 4.014483419548509e-12, 
             3.4487008515413167e-12, 2.9599834408260634e-12, 2.5382274687498563e-12, 2.174597613306697e-12, 
             1.861375526803235e-12, 1.5918254412904438e-12, 1.3600750230678129e-12, 1.1610098663476446e-12, 
             9.901801709822935e-13, 8.437182909232526e-13};
    '''


def produce_reduced_ramani_model(lineshape_method_name):
    """Create the model template of a reduced Ramani model which will use the given line shape.

    This produces composite models with the naming scheme "QMT_ReducedRamani_<lineshape_method_name>".

    Args:
        lineshape_method_name (str): the name of the lineshape method to use, one of
            Gaussian, Lorentzian, SuperLorentzian, GaussianInterpolated

    Returns:
        CompositeModelTemplate: the model template for the reduced Ramani model using the given line shape.
    """
    def extra_output_maps(results):
        np.warnings.filterwarnings('ignore')

        Rb, RM0a, fterm, gM0a, T1a_T2a, s0 = [
            np.squeeze(results[n]) for n in ('QMT.Rb', 'QMT.RM0a', 'QMT.fterm', 'QMT.gM0a', 'QMT.T1a_T2a', 'S0.s0')]
        Rb_std, RM0a_std, fterm_std, gM0a_std, T1a_T2a_std, s0_std = [
            np.squeeze(results.get(n + '.std', 0)) for n in ('QMT.Rb', 'QMT.RM0a', 'QMT.fterm',
                                                             'QMT.gM0a', 'QMT.T1a_T2a', 'S0.s0')]

        R_obs = 1. / results.get_input_data('T1')
        if len(R_obs.shape) > 1:
            R_obs = R_obs[:, 0]

        Ra = R_obs / (1.0 + ((RM0a * fterm * (Rb - R_obs)) / (Rb - R_obs + RM0a)))
        f = fterm * Ra / (1.0 + fterm * Ra)

        covars = create_covariance_matrix(
            results, ['QMT.RM0a', 'QMT.fterm', 'QMT.Rb'], results.get('covariances', None))

        Ra_gradient = get_ra_gradient(R_obs, RM0a, fterm, Rb)
        Ra_std = np.nan_to_num(np.sqrt(voxelwise_vector_matrix_vector_product(Ra_gradient, covars, Ra_gradient)))

        f_gradient = get_f_gradient(R_obs, RM0a, fterm, Rb)
        f_std = np.nan_to_num(np.sqrt(voxelwise_vector_matrix_vector_product(f_gradient, covars, f_gradient)))

        values = {
            'QMT.PD': gM0a * s0,
            'QMT.T1_f': 1.0 / Ra,
            'QMT.T2_f': (1.0 / Ra) / T1a_T2a,
            'QMT.k_bf': RM0a * f / (1.0 - f),
            'QMT.f_b': f,
            'QMT.Ra': Ra,
        }

        values.update({
            'QMT.PD.std': np.nan_to_num(np.sqrt(gM0a**2 * s0_std + s0**2 * gM0a_std)),
            'QMT.T1_f.std': np.nan_to_num(np.sqrt(Ra_std/Ra**4)),
            'QMT.T2_f.std': np.nan_to_num(np.sqrt((Ra**2 * T1a_T2a_std + Ra_std * T1a_T2a**2)/(Ra**4 * T1a_T2a**4))),
            'QMT.k_bf.std': np.nan_to_num(np.sqrt((RM0a**2 * f_std + RM0a_std * f**2 * (f - 1.0)**2) / (f - 1.0)**4)),
            'QMT.f_b.std': f_std,
            'QMT.Ra.std': Ra_std,
        })
        return values

    def get_ra_gradient(R_obs, RM0a, fterm, Rb):
        """Compute the standard deviation of Ra with respect to RM0a, fterm, Rb"""
        return np.column_stack([
            -R_obs*fterm*(R_obs - Rb)**2/(RM0a*fterm*(R_obs - Rb) - 1.0*RM0a + 1.0*R_obs - 1.0*Rb)**2,
            RM0a*R_obs*(R_obs - Rb)*(RM0a - R_obs + Rb)/(RM0a*fterm*(R_obs - Rb) - 1.0*RM0a + 1.0*R_obs - 1.0*Rb)**2,
            -RM0a**2*R_obs*fterm/(RM0a*fterm*(R_obs - Rb) - 1.0*RM0a + 1.0*R_obs - 1.0*Rb)**2])

    def get_f_gradient(R_obs, RM0a, fterm, Rb):
        """Compute the standard deviation of f with respect to RM0a, fterm, Rb"""
        return np.column_stack([
            1.0 * R_obs * fterm**2 * (R_obs - Rb)**2 * (-RM0a * fterm * (R_obs - Rb) + RM0a - R_obs + Rb) / (
                    (RM0a * fterm * (R_obs - Rb) - 1.0 * RM0a + 1.0 * R_obs - 1.0 * Rb) * (
                        -1.0 * RM0a * fterm * (R_obs - Rb) + 1.0 * RM0a + R_obs * fterm * (
                            RM0a - R_obs + Rb) - 1.0 * R_obs + 1.0 * Rb)**2),
        R_obs * (RM0a - R_obs + Rb) * (-RM0a * fterm * (R_obs - Rb) * (
                    -1.0 * RM0a * fterm * (R_obs - Rb) + 1.0 * RM0a + R_obs * fterm * (
                        RM0a - R_obs + Rb) - 1.0 * R_obs + 1.0 * Rb) + 1.0 * R_obs * fterm * (
                                                   RM0a - R_obs + Rb)**2 + (RM0a * fterm * (
                    R_obs - Rb) - 1.0 * RM0a + 1.0 * R_obs - 1.0 * Rb) * (
                                                   -1.0 * RM0a * fterm * (R_obs - Rb) + 1.0 * RM0a + R_obs * fterm * (
                                                       RM0a - R_obs + Rb) - 1.0 * R_obs + 1.0 * Rb)) / (
                    (RM0a * fterm * (R_obs - Rb) - 1.0 * RM0a + 1.0 * R_obs - 1.0 * Rb) * (
                        -1.0 * RM0a * fterm * (R_obs - Rb) + 1.0 * RM0a + R_obs * fterm * (
                            RM0a - R_obs + Rb) - 1.0 * R_obs + 1.0 * Rb)**2),
        1.0 * RM0a**2 * R_obs * fterm**2 * (-RM0a * fterm * (R_obs - Rb) + RM0a - R_obs + Rb) / (
                    (RM0a * fterm * (R_obs - Rb) - 1.0 * RM0a + 1.0 * R_obs - 1.0 * Rb) * (
                        -1.0 * RM0a * fterm * (R_obs - Rb) + 1.0 * RM0a + R_obs * fterm * (
                            RM0a - R_obs + Rb) - 1.0 * R_obs + 1.0 * Rb)**2)])

    class QMT(CompositeModelTemplate):
        name = 'QMT_ReducedRamani_' + lineshape_method_name
        model_expression = 'S0 * QMT(QMT)'
        likelihood_function = 'Gaussian'
        fixes = {'QMT.Rb': 1}
        extra_optimization_maps = [extra_output_maps]

        class QMT(CompartmentTemplate):
            parameters = ('Rb', 'RM0a', 'fterm', 'T2_b', 'T1a_T2a', 'gM0a',
                          'f0', 'B1', 'TR', 'sat_f0', 'sat_angle', 'pulse_Trf', 'pulse_p1', 'pulse_p2')
            dependencies = ['Lineshape' + lineshape_method_name]
            cl_code = '''
                double lsv = Lineshape''' + lineshape_method_name + '''(sat_f0 + f0, T2_b);
                
                double w_cwpe = (B1 * sat_angle / pulse_p1) * sqrt(pulse_p2 / (pulse_Trf * TR));
                double R_rfb = M_PI * (w_cwpe * w_cwpe) * lsv;
        
                double S = gM0a * (Rb * RM0a * fterm + R_rfb + Rb + RM0a) /
                    ((RM0a * fterm) * (Rb + R_rfb) + (1.0 + pown(w_cwpe / 
                        (2 * M_PI * sat_f0), 2) * T1a_T2a) * (R_rfb + Rb + RM0a));
        
                return S;
            '''

            class Rb(FreeParameterTemplate):
                init_value = 1
                lower_bound = 0
                upper_bound = 10
                numdiff_info = {'use_upper_bound': False}

            class RM0a(FreeParameterTemplate):
                """R*M0a"""
                init_value = 10
                lower_bound = 2
                upper_bound = 100
                numdiff_info = {'scale_factor': 0.1, 'use_upper_bound': False, 'use_lower_bound': False}

            class fterm(FreeParameterTemplate):
                """f/(R_a*(1-f))"""
                init_value = 0.1
                lower_bound = 1e-12
                upper_bound = 1
                numdiff_info = {'use_upper_bound': False}

            class T2_b(FreeParameterTemplate):
                """T2_b"""
                init_value = 10e-6
                lower_bound = 1e-7
                upper_bound = 100e-6
                parameter_transform = ScaleTransform(1e6)
                numdiff_info = {'scale_factor': 1e5, 'use_upper_bound': False, 'use_lower_bound': False}

            class T1a_T2a(FreeParameterTemplate):
                """T1_a/T2_a"""
                init_value = 25
                lower_bound = 1
                upper_bound = 50
                numdiff_info = {'use_upper_bound': False}

            class gM0a(FreeParameterTemplate):
                """gM0_a"""
                init_value = 1
                lower_bound = 0.5
                upper_bound = 1.5
                numdiff_info = {'use_lower_bound': False, 'use_upper_bound': False}

            class f0(ProtocolParameterTemplate):
                pass

            class B1(ProtocolParameterTemplate):
                pass

            class TR(ProtocolParameterTemplate):
                pass

            class sat_f0(ProtocolParameterTemplate):
                pass

            class sat_angle(ProtocolParameterTemplate):
                pass

            class pulse_Trf(ProtocolParameterTemplate):
                pass

            class pulse_p1(ProtocolParameterTemplate):
                pass

            class pulse_p2(ProtocolParameterTemplate):
                pass


produce_reduced_ramani_model('Gaussian')
produce_reduced_ramani_model('Lorentzian')
produce_reduced_ramani_model('SuperLorentzian')
produce_reduced_ramani_model('SuperLorentzianInterpolated')
