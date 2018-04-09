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
