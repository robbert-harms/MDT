from mdt import LibraryFunctionTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TensorApparentDiffusion(LibraryFunctionTemplate):

    description = '''
        Calculates the apparent diffusion for the Tensor model.
        
        Args:
            theta: polar angle of the first vector
            phi: azimuth angle of the first vector
            psi: rotation around the first vector, used to generate the perpendicular vectors.
            d: first eigenvalue
            dperp0: second eigenvalue
            dperp1: third eigenvalue
    '''
    return_type = 'mot_float_type'
    dependencies = ['TensorSphericalToCartesian']
    parameters = ['theta', 'phi', 'psi', 'd', 'dperp0', 'dperp1', 'g']
    cl_code = '''
        mot_float_type4 vec0, vec1, vec2;
        TensorSphericalToCartesian(theta, phi, psi, &vec0, &vec1, &vec2);
        
        return  d *      pown(dot(vec0, g), 2) +
                dperp0 * pown(dot(vec1, g), 2) +
                dperp1 * pown(dot(vec2, g), 2);
    '''
