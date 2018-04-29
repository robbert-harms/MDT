from mot.cl_routines.mapping.codec_runner import CodecRunner
from mot.model_interfaces import OptimizeModelInterface
from mot.utils import NameFunctionTuple


__author__ = 'Robbert Harms'
__date__ = '2017-05-29'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class ParameterCodec(object):

    def get_parameter_encode_function(self, fname='encodeParameters'):
        """Get a CL function that can transform the model parameters from model space to an encoded space.

        The signature of the CL function is:

        .. code-block:: c

            void <fname>(mot_data_struct* data, mot_float_type* x);

        Args:
            fname (str): The CL function name to use

        Returns:
            str: An OpenCL function that is used in the CL kernel to transform the parameters from model space to
                encoded space so they can be used as input to an CL routine.
        """
        raise NotImplementedError()

    def get_parameter_decode_function(self, fname='decodeParameters'):
        """Get a CL function that can transform the model parameters from encoded space to model space.

        The signature of the CL function is:

        .. code-block:: c

            void <fname>(mot_data_struct* data, mot_float_type* x);

        Args:
            fname (str): The CL function name to use

        Returns:
            str: An OpenCL function that is used in the CL kernel to transform the parameters from encoded space to
                model space so they can be used as input to the model.
        """
        raise NotImplementedError()


class ParameterTransformedModel(OptimizeModelInterface):

    def __init__(self, model, parameter_codec):
        """Decorates the given model with parameter encoding and decoding transformations.

        This decorates a few of the given function calls with the right parameter encoding and decoding transformations
        such that both the underlying model and the calling routines are unaware that the parameters have been altered.

        Args:
            model (OptimizeModelInterface): the model to decorate
            parameter_codec (mdt.model_building.utils.ParameterCodec): the parameter codec to use
        """
        self._model = model
        self._parameter_codec = parameter_codec

    def decode_parameters(self, parameters):
        """Decode the given parameters back to model space.

        Args:
            parameters (ndarray): the parameters to transform back to model space
        """
        space_transformer = CodecRunner()
        return space_transformer.decode(parameters, self.get_kernel_data(), self._parameter_codec)

    def encode_parameters(self, parameters):
        """Decode the given parameters into optimization space

        Args:
            parameters (ndarray): the parameters to transform into optimization space
        """
        space_transformer = CodecRunner()
        return space_transformer.encode(parameters, self.get_kernel_data(), self._parameter_codec)

    def get_kernel_data(self):
        return self._model.get_kernel_data()

    def get_nmr_problems(self):
        return self._model.get_nmr_problems()

    def get_nmr_observations(self):
        return self._model.get_nmr_observations()

    def get_nmr_parameters(self):
        return self._model.get_nmr_parameters()

    def get_pre_eval_parameter_modifier(self):
        old_modifier = self._model.get_pre_eval_parameter_modifier()
        new_fname = 'wrapped_' + old_modifier.get_cl_function_name()

        code = old_modifier.get_cl_code()
        code += self._parameter_codec.get_parameter_decode_function('_decodeParameters')
        code += '''
            void ''' + new_fname + '''(mot_data_struct* data, mot_float_type* x){
                _decodeParameters(data, x);
                ''' + old_modifier.get_cl_function_name() + '''(data, x);
            }
        '''
        return NameFunctionTuple(new_fname, code)

    def get_objective_per_observation_function(self):
        return self._model.get_objective_per_observation_function()

    def get_lower_bounds(self):
        # todo add codec transform here
        return self._model.get_lower_bounds()

    def get_upper_bounds(self):
        # todo add codec transform here
        return self._model.get_upper_bounds()

    def finalize_optimized_parameters(self, parameters):
        return self._model.finalize_optimized_parameters(self.decode_parameters(parameters))

    def __getattr__(self, item):
        return getattr(self._model, item)
