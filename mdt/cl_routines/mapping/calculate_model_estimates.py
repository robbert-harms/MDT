import pyopencl as cl
import numpy as np
from mot.utils import get_float_type_def, KernelDataManager
from mot.cl_routines.base import CLRoutine
from mot.load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CalculateModelEstimates(CLRoutine):

    def calculate(self, model, parameters):
        """Evaluate the model for every problem and every observation and return the estimates.

        This evaluates only the model and not the likelihood function of the model given the measured data.

        Args:
            model (AbstractModel): The model to evaluate.
            parameters (ndarray): The parameters to use in the evaluation of the model

        Returns:
            ndarray: Return per problem instance the evaluation per data point.
        """
        nmr_observations = model.get_nmr_observations()

        parameters = np.require(parameters, self._cl_runtime_info.mot_float_dtype, requirements=['C', 'A', 'O'])

        nmr_problems = parameters.shape[0]
        evaluations = np.zeros((nmr_problems, nmr_observations),
                               dtype=self._cl_runtime_info.mot_float_dtype, order='C')

        workers = self._create_workers(lambda cl_environment: _EvaluateModelWorker(
            cl_environment, self._cl_runtime_info.get_compile_flags(), model, parameters, evaluations,
            self._cl_runtime_info.mot_float_dtype, self._cl_runtime_info.double_precision))
        self._cl_runtime_info.load_balancer.process(workers, nmr_problems)

        return evaluations


class _EvaluateModelWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model, parameters, evaluations, mot_float_dtype,
                 double_precision):
        super(_EvaluateModelWorker, self).__init__(cl_environment)

        self._model = model
        self._data_info = self._model.get_kernel_data()
        self._data_struct_manager = KernelDataManager(self._data_info, mot_float_dtype)
        self._double_precision = double_precision
        self._evaluations = evaluations
        self._parameters = parameters

        self._all_buffers, self._evaluations_buffer = self._create_buffers()
        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        kernel_func = self._kernel.get_estimates

        scalar_args = [None, None]
        scalar_args.extend(self._data_struct_manager.get_scalar_arg_dtypes())
        kernel_func.set_scalar_arg_dtypes(scalar_args)

        kernel_func(self._cl_queue, (int(nmr_problems), ), None,
                    *self._all_buffers, global_offset=(int(range_start),))
        self._enqueue_readout(self._evaluations_buffer, self._evaluations, range_start, range_end)

    def _create_buffers(self):
        evaluations_buffer = cl.Buffer(self._cl_context,
                                       cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                       hostbuf=self._evaluations)

        all_buffers = [cl.Buffer(self._cl_context,
                                 cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                 hostbuf=self._parameters),
                       evaluations_buffer]
        all_buffers.extend(self._data_struct_manager.get_kernel_inputs(self._cl_context, 1))
        return all_buffers, evaluations_buffer

    def _get_kernel_source(self):
        eval_function_info = self._model.get_model_eval_function()
        param_modifier = self._model.get_pre_eval_parameter_modifier()

        nmr_params = self._parameters.shape[1]

        kernel_param_names = ['global mot_float_type* restrict params',
                              'global mot_float_type* restrict estimates']
        kernel_param_names.extend(self._data_struct_manager.get_kernel_arguments())

        kernel_source = '''
            #define NMR_OBSERVATIONS ''' + str(self._model.get_nmr_observations()) + '''
        '''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._data_struct_manager.get_struct_definition()
        kernel_source += eval_function_info.get_cl_code()
        kernel_source += param_modifier.get_cl_code()
        kernel_source += '''
            __kernel void get_estimates(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ulong gid = get_global_id(0);
                    mot_data_struct data = ''' + self._data_struct_manager.get_struct_init_string('gid') + ''';

                    mot_float_type x[''' + str(nmr_params) + '''];
                    for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    global mot_float_type* result = estimates + gid * NMR_OBSERVATIONS;
                    
                    ''' + param_modifier.get_cl_function_name() + '''(&data, x);
                    
                    for(uint i = 0; i < NMR_OBSERVATIONS; i++){
                        result[i] = ''' + eval_function_info.get_cl_function_name() + '''(&data, x, i);
                    }
            }
        '''
        return kernel_source
