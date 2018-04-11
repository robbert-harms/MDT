from mdt.model_building.model_builders import ModelBuilder
from mot.model_interfaces import SampleModelInterface, NumericalDerivativeInterface, OptimizeModelInterface

__author__ = 'Robbert Harms'
__date__ = '2017-08-31'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class MRIModelBuilder(ModelBuilder):
    """This extends the :class:`~mdt.model_building.model_builders.ModelBuilder` interface with some extra functions.

    The model fitting in MDT requires some extra functions, both of the model builder as of the build model.
    """

    def build(self, problems_to_analyze=None):
        """Construct the final model using all current construction settings.

        Args:
            problems_to_analyze (ndarray): optional set of problem indices, this should construct the model
                such that it analyzes only the indicated subset of problems.

        Returns:
            MRIModelInterface: the MRI build model
        """
        raise NotImplementedError()


class MRIModelInterface(SampleModelInterface, NumericalDerivativeInterface):
    """Extends the :class:`~mot.model_interfaces.SampleModelInterface` for use within MDT."""

    def get_post_optimization_output(self, optimization_results):
        """Transform the optimization results into a dictionary with all maps we would like to save.

        This is where additional post processing can take place on the optimization results.

        Args:
            optimization_results (mot.cl_routines.optimizing.base.OptimizationResults): the result object from
                the optimizer.

        Returns:
            dict: the output dictionary. Every value of this dictionary can be ndarray or a dictionary. If it is a
                dictionary we create a subfolder with the key of that value and store all items in that sub-dictionary
                in that subfolder.

                For example, the dictionary ``{'a': ndarray1, 'b': {'c': ndarray2}}`` would write the files:

                    * ./a.nii.gz
                    * ./b/c.nii.gz
        """
        raise NotImplementedError()
