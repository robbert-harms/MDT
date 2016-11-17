from mdt.models.compartments import CompartmentConfig

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SphereGPD(CompartmentConfig):

    parameter_list = ('Delta', 'delta', 'd', 'R')
    dependency_list = ('MRIConstants',)
