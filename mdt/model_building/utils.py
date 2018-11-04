from mot.lib.cl_function import SimpleCLFunction
from mot.lib.kernel_data import Array, Struct, LocalMemory

__author__ = 'Robbert Harms'
__date__ = '2017-05-29'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class ObjectiveFunctionWrapper:

    def __init__(self, nmr_parameters):
        self._nmr_parameters = nmr_parameters

    def wrap_input_data(self, input_data):
        """Wrap the input data with extra information this wrapper might need.

        Args:
            input_data (mot.lib.kernel_data.KernelData): the kernel data we will wrap

        Returns:
            mot.lib.kernel_data.KernelData: the wrapped kernel data
        """
        return Struct({'data': input_data, 'x_tmp': LocalMemory('mot_float_type', nmr_items=self._nmr_parameters)},
                      'objective_function_wrapper_data')

    def wrap_objective_function(self, objective_function, decode_function):
        """Decorates the given objective function with parameter decoding.

        This will change the given parameter vector in-place. This is possible because the optimization routines
        make a copy of the vector before handing it over to the optimization function.

        Args:
            objective_function (mot.lib.cl_function.CLFunction): A CL function with the signature:

                .. code-block:: c

                    double <func_name>(local mot_float_type* x,
                                       void* data,
                                       local mot_float_type* objective_list);
            decode_function (mot.lib.cl_function.CLFunction): An OpenCL function that is used in the CL kernel to
                    transform the parameters from encoded space to model space so they can be used as input to the model.
                    The signature of the CL function is:

                    .. code-block:: c

                        void <fname>(void* data, local mot_float_type* x);

            nmr_parameters (int): the number of parameters we are decoding.

        Returns:
            mot.lib.cl_function.CLFunction: the wrapped objective function.
        """
        return SimpleCLFunction.from_string('''
            double wrapped_''' + objective_function.get_cl_function_name() + '''(
                    local mot_float_type* x,
                    void* data, 
                    local mot_float_type* objective_list){
                
                local mot_float_type* x_tmp = ((objective_function_wrapper_data*)data)->x_tmp;
                
                if(get_local_id(0) == 0){
                    for(uint i = 0; i < ''' + str(self._nmr_parameters) + '''; i++){
                        x_tmp[i] = x[i];
                    }
                    ''' + decode_function.get_cl_function_name() + '''(
                        ((objective_function_wrapper_data*)data)->data, 
                        x_tmp);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
    
                return ''' + objective_function.get_cl_function_name() + '''(
                    x_tmp, ((objective_function_wrapper_data*)data)->data, objective_list);    
            }
        ''', dependencies=[objective_function, decode_function])


class ParameterCodec:

    def __init__(self, encode_func, decode_func, encode_bounds_func=None):
        """Create a parameter codec container.

        Args:
            encode_func (mot.lib.cl_function.CLFunction): An OpenCL function that is used in the CL kernel to
                transform the parameters from model space to encoded space so they can be used as input to an
                CL routine. The signature of the CL function is:

                .. code-block:: c

                    void <fname>(void* data, local mot_float_type* x);

            decode_func (mot.lib.cl_function.CLFunction): An OpenCL function that is used in the CL kernel to
                transform the parameters from encoded space to model space so they can be used as input to the model.
                The signature of the CL function is:

                .. code-block:: c

                    void <fname>(void* data, local mot_float_type* x);
            encode_bounds_func (Callable[[array, array], Tuple[array, array]]): encode the lower and upper bounds
                to bounds of the encoded parameter space. If not set, we won't encode the bounds
        """
        self._encode_func = encode_func
        self._decode_func = decode_func
        self._encode_bounds_func = encode_bounds_func

    def get_encode_function(self):
        """Get a CL function that can transform the model parameters from model space to an encoded space.

        Returns:
            mot.lib.cl_function.CLFunction: An OpenCL function that is used in the CL kernel to transform the parameters
                from model space to encoded space so they can be used as input to an CL routine.
                The signature of the CL function is:

                .. code-block:: c

                    void <fname>(void* data, local mot_float_type* x);
        """
        return self._encode_func

    def get_decode_function(self):
        """Get a CL function that can transform the model parameters from encoded space to model space.

        Returns:
            mot.lib.cl_function.CLFunction: An OpenCL function that is used in the CL kernel to transform the parameters
                from encoded space to model space so they can be used as input to the model.
                The signature of the CL function is:

                .. code-block:: c

                    void <fname>(void* data, local mot_float_type* x);
        """
        return self._decode_func

    def encode_bounds(self, lower_bounds, upper_bounds):
        """Encode the given bounds to the encoded parameter space.

        Args:
            lower_bounds (list): for each parameter the lower bound(s). Each element can either be a scalar or a vector.
            upper_bounds (list): for each parameter the upper bound(s). Each element can either be a scalar or a vector.

        Returns:
            tuple: the lower and the upper bounds, in a similar structure as the input
        """
        if self._encode_bounds_func is not None:
            return self._encode_bounds_func(lower_bounds, upper_bounds)
        return lower_bounds, upper_bounds

    def decode(self, parameters, kernel_data=None, cl_runtime_info=None):
        """Decode the given parameters using the given model.

        This transforms the data from optimization space to model space.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.lib.utils.KernelData]): the additional data to load
            cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information

        Returns:
            ndarray: The array with the transformed parameters.
        """
        return self._transform_parameters(self.get_decode_function(),
                                          parameters, kernel_data, cl_runtime_info=cl_runtime_info)

    def encode(self, parameters, kernel_data=None, cl_runtime_info=None):
        """Encode the given parameters using the given model.

        This transforms the data from model space to optimization space.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.lib.utils.KernelData]): the additional data to load
            cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information

        Returns:
            ndarray: The array with the transformed parameters.
        """
        return self._transform_parameters(self.get_encode_function(),
                                          parameters, kernel_data, cl_runtime_info=cl_runtime_info)

    def encode_decode(self, parameters, kernel_data=None, cl_runtime_info=None):
        """First apply an encoding operation and then apply a decoding operation again.

        This can be used to enforce boundary conditions in the parameters.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.lib.utils.KernelData]): the additional data to load
            cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information

        Returns:
            ndarray: The array with the transformed parameters.
        """
        encode_func = self.get_encode_function()
        decode_func = self.get_decode_function()

        func = SimpleCLFunction.from_string('''
            void encode_decode_parameters(void* data, local mot_float_type* x){
                ''' + encode_func.get_cl_function_name() + '''(data, x);
                ''' + decode_func.get_cl_function_name() + '''(data, x);
            }
        ''', dependencies=[encode_func, decode_func])
        return self._transform_parameters(func, parameters, kernel_data, cl_runtime_info=cl_runtime_info)

    @staticmethod
    def _transform_parameters(cl_func, parameters, kernel_data, cl_runtime_info=None):
        cl_named_func = SimpleCLFunction.from_string('''
            void transformParameterSpace(void* data, local mot_float_type* x){
                ''' + cl_func.get_cl_function_name() + '''(data, x);
            }
        ''', dependencies=[cl_func])

        kernel_data = {'data': kernel_data,
                       'x': Array(parameters, ctype='mot_float_type', mode='rw')}

        cl_named_func.evaluate(kernel_data, parameters.shape[0], cl_runtime_info=cl_runtime_info)
        return kernel_data['x'].get_data()
