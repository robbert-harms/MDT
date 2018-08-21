from mot.lib.cl_function import SimpleCLFunction
from mot.lib.kernel_data import Array


__author__ = 'Robbert Harms'
__date__ = '2017-05-29'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


def wrap_objective_function(objective_function, decode_function, nmr_parameters):
    """Decorates the given objective function with parameter decoding.

    Args:
        objective_function (mot.lib.cl_function.CLFunction): A CL function with the signature:

            .. code-block:: c

                double <func_name>(local const mot_float_type* const x,
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
                local const mot_float_type* const x,
                void* data, 
                local mot_float_type* objective_list){

            local mot_float_type x_model[''' + str(nmr_parameters) + '''];

            if(get_local_id(0) == 0){
                for(uint i = 0; i < ''' + str(nmr_parameters) + '''; i++){
                    x_model[i] = x[i];
                }
                ''' + decode_function.get_cl_function_name() + '''(data, x_model);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            return ''' + objective_function.get_cl_function_name() + '''(
                x_model, data, objective_list);    
        }
    ''', dependencies=[objective_function, decode_function])


class ParameterCodec(object):

    def __init__(self, encode_func, decode_func):
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
        """
        self._encode_func = encode_func
        self._decode_func = decode_func

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

    def encode_decode(self, parameters, kernel_data, codec, cl_runtime_info=None):
        """First apply an encoding operation and then apply a decoding operation again.

        This can be used to enforce boundary conditions in the parameters.

        Args:
            parameters (ndarray): The parameters to transform
            kernel_data (dict[str: mot.lib.utils.KernelData]): the additional data to load
            cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information

        Returns:
            ndarray: The array with the transformed parameters.
        """
        encode_func = codec.get_encode_function()
        decode_func = codec.get_decode_function()

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
