from pkg_resources import resource_filename
from mdt.model_parameters import get_parameter
from mdt.utils import DMRICompartmentModelFunction
from mdt.cl_routines.mapping.dti_measures import DTIMeasures
from mdt.utils import eigen_vectors_from_tensor
from mot import runtime_configuration
from mot.parameter_functions.transformations import SinSqrClampDependentTransform
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


meta_info = {'description': 'The Tensor compartment model.'}


class Tensor(DMRICompartmentModelFunction):

    def __init__(self, name='Tensor'):
        super(Tensor, self).__init__(
            name,
            'cmTensor',
            (get_parameter('g'),
             get_parameter('b'),
             get_parameter('d'),
             get_parameter('dperp0'),
             get_parameter('dperp1'),
             get_parameter('theta'),
             get_parameter('phi'),
             get_parameter('psi')),
            resource_filename(__name__, 'Tensor.h'),
            resource_filename(__name__, 'Tensor.cl'),
            ()
        )

        self.get_parameter_by_name('dperp0').parameter_transform = \
            SinSqrClampDependentTransform(((self, self.get_parameter_by_name('d')),))

        self.get_parameter_by_name('dperp1').parameter_transform = \
            SinSqrClampDependentTransform(((self, self.get_parameter_by_name('dperp0')),))

    def get_extra_results_maps(self, results_dict):
        """This will return the eigenvectors and values for the Tensor.

        It should return the vectors and values sorted on eigenvalues from large to small. We here assume that
        in the results dictionary the following holds: d > dperp0 > dperp1.
        """
        eigen_vectors = eigen_vectors_from_tensor(results_dict[self.name + '.theta'], results_dict[self.name + '.phi'],
                                                  results_dict[self.name + '.psi'])

        eigen_values = [results_dict[self.name + '.d'], results_dict[self.name + '.dperp0'],
                        results_dict[self.name + '.dperp1']]

        extra_maps = {}
        for ind in range(3):
            extra_maps.update({self.name + '.eig' + repr(ind) + '.vec': eigen_vectors[:, ind],
                               self.name + '.eig' + repr(ind) + '.val': eigen_values[ind]})

            for dimension in range(3):
                extra_maps.update({self.name + '.eig' + repr(ind) + '.vec.' + repr(dimension):
                                   eigen_vectors[:, ind, dimension]})

        fa, md = DTIMeasures(runtime_configuration.runtime_config['cl_environments'],
                             runtime_configuration.runtime_config['load_balancer']).\
            concat_and_calculate(eigen_values[0], eigen_values[1], eigen_values[2])
        extra_maps.update({self.name + '.FA': fa, self.name + '.MD': md})

        extra_maps.update({self.name + '.AD': eigen_values[0]})
        extra_maps.update({self.name + '.RD': (eigen_values[1] + eigen_values[2]) / 2})

        return extra_maps