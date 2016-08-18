from mdt.models.compartments import CompartmentConfig


class SSFP_Stick(CompartmentConfig):

    parameter_list = ('g', 'd', 'TR', 'flip_angle', 'b1map', 'T1map', 'T2map')
