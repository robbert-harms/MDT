from mot.lib.cl_function import SimpleCLFunction
from mot.lib.kernel_data import Array
from mot.lib.model_interfaces import OptimizeModelInterface


__author__ = 'Robbert Harms'
__date__ = '2017-05-29'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class ParameterCodec(object):

    def get_parameter_encode_function(self):
        """Get a CL function that can transform the model parameters from model space to an encoded space.

        Returns:
            mot.lib.cl_function.CLFunction: An OpenCL function that is used in the CL kernel to transform the parameters
                from model space to encoded space so they can be used as input to an CL routine.
                The signature of the CL function is:

                .. code-block:: c

                    void <fname>(mot_data_struct* data, local mot_float_type* x);
        """
        raise NotImplementedError()

    def get_parameter_decode_function(self):
        """Get a CL function that can transform the model parameters from encoded space to model space.

        Returns:
            mot.lib.cl_function.CLFunction: An OpenCL function that is used in the CL kernel to transform the parameters
                from encoded space to model space so they can be used as input to the model.
                The signature of the CL function is:

                .. code-block:: c

                    void <fname>(mot_data_struct* data, local mot_float_type* x);
        """
        raise NotImplementedError()

    def decode(self, parameters, kernel_data, cl_runtime_info=None):
        """Decode the given parameters using the given model.

        This transforms the data from optimization space to model space.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.lib.utils.KernelData]): the additional data to load
            cl_runtime_info (mot.lib.cl_runtime_info.CLRuntimeInfo): the runtime information

        Returns:
            ndarray: The array with the transformed parameters.
        """
        return self._transform_parameters(self.get_parameter_decode_function(),
                                          parameters, kernel_data, cl_runtime_info=cl_runtime_info)

    def encode(self, parameters, kernel_data, cl_runtime_info=None):
        """Encode the given parameters using the given model.

        This transforms the data from model space to optimization space.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.lib.utils.KernelData]): the additional data to load
            cl_runtime_info (mot.lib.cl_runtime_info.CLRuntimeInfo): the runtime information

        Returns:
            ndarray: The array with the transformed parameters.
        """
        return self._transform_parameters(self.get_parameter_encode_function(),
                                          parameters, kernel_data, cl_runtime_info=cl_runtime_info)

    def encode_decode(self, parameters, kernel_data, codec, cl_runtime_info=None):
        """First apply an encoding operation and then apply a decoding operation again.

        This can be used to enforce boundary conditions in the parameters.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.lib.utils.KernelData]): the additional data to load
            cl_runtime_info (mot.lib.cl_runtime_info.CLRuntimeInfo): the runtime information

        Returns:
            ndarray: The array with the transformed parameters.
        """
        encode_func = codec.get_parameter_encode_function()
        decode_func = codec.get_parameter_decode_function()

        func = SimpleCLFunction.from_string('''
            void encode_decode_parameters(mot_data_struct* data, local mot_float_type* x){
                ''' + encode_func.get_cl_function_name() + '''(data, x);
                ''' + decode_func.get_cl_function_name() + '''(data, x);
            }
        ''', dependencies=[encode_func, decode_func])
        return self._transform_parameters(func, parameters, kernel_data, cl_runtime_info=cl_runtime_info)

    def _transform_parameters(self, cl_func, parameters, kernel_data, cl_runtime_info=None):
        cl_named_func = SimpleCLFunction.from_string('''
            void transformParameterSpace(mot_data_struct* data){
                local mot_float_type x[''' + str(parameters.shape[1]) + '''];
                for(uint i = 0; i < ''' + str(parameters.shape[1]) + '''; i++){
                    x[i] = data->x[i];
                }

                ''' + cl_func.get_cl_function_name() + '''(data, x);

                for(uint i = 0; i < ''' + str(parameters.shape[1]) + '''; i++){
                    data->x[i] = x[i];
                }
            }
        ''', dependencies=[cl_func])

        data_struct = dict(kernel_data)
        data_struct['x'] = Array(parameters, ctype='mot_float_type', is_writable=True)

        cl_named_func.evaluate({'data': data_struct}, nmr_instances=parameters.shape[0],
                               cl_runtime_info=cl_runtime_info)
        return data_struct['x'].get_data()


class ParameterTransformedModel(OptimizeModelInterface):

    def __init__(self, model, parameter_codec, nmr_parameters):
        """Decorates the given model with parameter encoding and decoding transformations.

        This decorates a few of the given function calls with the right parameter encoding and decoding transformations
        such that both the underlying model and the calling routines are unaware that the parameters have been altered.

        Args:
            model (OptimizeModelInterface): the model to decorate
            parameter_codec (mdt.model_building.utils.ParameterCodec): the parameter codec to use
            nmr_parameters (int): the number of parameters in the model
        """
        self._model = model
        self._parameter_codec = parameter_codec
        self._nmr_parameters = nmr_parameters

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

    def get_nmr_observations(self):
        return self._model.get_nmr_observations()

    def get_objective_function(self):
        objective_function = self._model.get_objective_function()
        decode_func = self._parameter_codec.get_parameter_decode_function()
        return SimpleCLFunction.from_string('''
            double wrapped_''' + objective_function.get_cl_function_name() + '''(
                    mot_data_struct* data, 
                    local const mot_float_type* const x,
                    local mot_float_type* objective_list,
                    local double* objective_value_tmp){
                
                local mot_float_type x_model[''' + str(self._nmr_parameters) + '''];
                
                if(get_local_id(0) == 0){
                    for(uint i = 0; i < ''' + str(self._nmr_parameters) + '''; i++){
                        x_model[i] = x[i];
                    }
                    ''' + decode_func.get_cl_function_name() + '''(data, x_model);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                
                return ''' + objective_function.get_cl_function_name() + '''(
                    data, x_model, objective_list, objective_value_tmp);    
            }
        ''', dependencies=[objective_function, decode_func])

    def get_lower_bounds(self):
        # todo add codec transform here
        return self._model.get_lower_bounds()

    def get_upper_bounds(self):
        # todo add codec transform here
        return self._model.get_upper_bounds()

    def __getattr__(self, item):
        return getattr(self._model, item)
