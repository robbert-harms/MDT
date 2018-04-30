from mdt.component_templates.library_functions import LibraryFunctionTemplate


__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NeumannCylinderRestrictedLongApprox(LibraryFunctionTemplate):

    description = '''
        This function returns the displacement in the restricted signal attenuation, for radius R according to the 
        long diffusion time approximation of the Von Neumann model, equation (28) in [1].
        
        In spin-echo experiments, tau is typically TE/2.
        
        References:
        1) 1. Neuman CH. Spin echo of spins diffusing in a bounded medium. 
            J Chem Phys. 1974;60(11):4508-4511. doi:10.1063/1.1680931.
    '''
    return_type = 'double'
    parameters = ['G', 'tau', 'd', 'R']
    dependencies = ['MRIConstants']
    cl_code = '''
        return -( (pown(R, 4) * GAMMA_H_HZ_SQ * (G * G)) / d ) 
                    * (7 / 296.0) 
                    * (2 * tau - (99/112.0) * (pown(R, 2) / d)); 
    '''
