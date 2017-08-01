from mdt.component_templates.library_functions import LibraryFunctionTemplate
from mot.cl_data_type import SimpleCLDataType
from mot.model_building.parameters import LibraryParameter

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class RotateVectors(LibraryFunctionTemplate):

    description = '''
        Uses Rodrigues' rotation formula to rotate the given vector v by psi around k.
        
        Args:
            basis: the unit vector defining the rotation axis (k)
            to_rotate: the vector to rotate by angle psi (v)
            psi: the rotation angle (psi)
        
        Returns:
            vector: the rotated vector
    '''
    return_type = 'mot_float_type4'
    parameter_list = [LibraryParameter('mot_float_type4', 'basis'),
                      LibraryParameter('mot_float_type4', 'to_rotate'),
                      LibraryParameter('mot_float_type', 'psi')]
    cl_code = '''
        mot_float_type cos_psi;
        mot_float_type sin_psi = sincos(psi, &cos_psi);
        
        // using a multiplication factor to prevent commutative problems in the cross product between the two vectors.
        char multiplication_factor = ((basis).z < 0 || (((basis).z == 0.0) && (basis).x < 0.0)) ? -1 : 1;
        
        return to_rotate * cos_psi
                + (basis * dot(basis, to_rotate) * (1-cos_psi)
                + (cross(to_rotate, basis * multiplication_factor) * sin_psi));
    '''
