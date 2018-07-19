from mot.cl_function import SimpleCLFunction
from mot.cl_routines.base import RunProcedure
from mot.kernel_data import KernelArray
from mot.model_interfaces import OptimizeModelInterface


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

    def decode(self, parameters, kernel_data, cl_runtime_info=None):
        """Decode the given parameters using the given model.

        This transforms the data from optimization space to model space.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.utils.KernelData]): the additional data to load
            cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the runtime information

        Returns:
            ndarray: The array with the transformed parameters.
        """
        return self._transform_parameters(self.get_parameter_decode_function('decodeParameters'),
                                          'decodeParameters', parameters, kernel_data, cl_runtime_info=cl_runtime_info)

    def encode(self, parameters, kernel_data, cl_runtime_info=None):
        """Encode the given parameters using the given model.

        This transforms the data from model space to optimization space.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.utils.KernelData]): the additional data to load
            cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the runtime information

        Returns:
            ndarray: The array with the transformed parameters.
        """
        return self._transform_parameters(self.get_parameter_encode_function('encodeParameters'),
                                          'encodeParameters', parameters, kernel_data, cl_runtime_info=cl_runtime_info)

    def encode_decode(self, parameters, kernel_data, codec, cl_runtime_info=None):
        """First apply an encoding operation and then apply a decoding operation again.

        This can be used to enforce boundary conditions in the parameters.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.utils.KernelData]): the additional data to load
            cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the runtime information

        Returns:
            ndarray: The array with the transformed parameters.
        """
        func_name = 'encode_decode'
        func = ''
        func += codec.get_parameter_encode_function('encodeParameters')
        func += codec.get_parameter_decode_function('decodeParameters')
        func += '''
            void ''' + func_name + '''(mot_data_struct* data, mot_float_type* x){
                encodeParameters(data, x);
                decodeParameters(data, x);
            }
        '''
        return self._transform_parameters(func, func_name, parameters, kernel_data, cl_runtime_info=cl_runtime_info)

    def _transform_parameters(self, cl_func, cl_func_name, parameters, kernel_data, cl_runtime_info=None):
        cl_named_func = self._get_codec_function_wrapper(cl_func, cl_func_name, parameters.shape[1])

        all_kernel_data = dict(kernel_data)
        all_kernel_data['x'] = KernelArray(parameters, ctype='mot_float_type', is_writable=True)

        runner = RunProcedure(cl_runtime_info)
        runner.run_procedure(cl_named_func, all_kernel_data, parameters.shape[0])
        return all_kernel_data['x'].get_data()

    def _get_codec_function_wrapper(self, cl_func, cl_func_name, nmr_params):
        cl_body = '''
            mot_float_type x[''' + str(nmr_params) + '''];
            for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                x[i] = data->x[i];
            }

            ''' + cl_func_name + '''(data, x);

            for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                data->x[i] = x[i];
            }
        '''
        return SimpleCLFunction('void', 'transformParameterSpace', [('mot_data_struct*', 'data')], cl_body,
                                cl_extra=cl_func)


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
        return self._parameter_codec.decode(parameters, self.get_kernel_data())

    def encode_parameters(self, parameters):
        """Decode the given parameters into optimization space

        Args:
            parameters (ndarray): the parameters to transform into optimization space
        """
        return self._parameter_codec.encode(parameters, self.get_kernel_data())

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
        body = '''
            _decodeParameters(data, x);
            ''' + old_modifier.get_cl_function_name() + '''(data, x);
        '''
        return SimpleCLFunction(
            'void', 'wrapped_' + old_modifier.get_cl_function_name(),
            [('mot_data_struct*', 'data'), ('mot_float_type*', 'x')], body,
            dependencies=[old_modifier],
            cl_extra=self._parameter_codec.get_parameter_decode_function('_decodeParameters'))

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
