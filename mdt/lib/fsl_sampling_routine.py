import numpy as np
from mot.sample.base import AbstractRWMSampler
from mot.lib.kernel_data import Array

__author__ = 'Robbert Harms'
__date__ = '2018-08-14'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class FSLSamplingRoutine(AbstractRWMSampler):

    def __init__(self, ll_func, log_prior_func, x0, proposal_stds, batch_size=50, min_val=1e-15, max_val=1e3, **kwargs):
        r"""An acceptance rate scaling algorithm found in a Neuroscience package called FSL.

        This scaling algorithm scales the std. by :math:`\sqrt(a/(n - a))` where a is the number of accepted samples
        in the last batch and n is the batch size. Its goal is to balance the acceptance rate at 0.5.

        Since this method never ceases the adaptation of the standard deviations, it theoretically loses
        ergodicity of the chain [1, 2].

        Args:
            ll_func (mot.lib.cl_function.CLFunction): The log-likelihood function. See parent docs.
            log_prior_func (mot.lib.cl_function.CLFunction): The log-prior function. See parent docs.
            x0 (ndarray): the starting positions for the sampler. Should be a two dimensional matrix
                with for every modeling instance (first dimension) and every parameter (second dimension) a value.
            proposal_stds (ndarray): for every parameter and every modeling instance an initial proposal std.
            batch_size (int): the size of the batches inbetween which we update the parameters
            min_val (float): the minimum value the standard deviation can take
            max_val (float): the maximum value the standard deviation can take

        References:
            [1] Roberts GO, Rosenthal JS. Examples of adaptive MCMC. J Comput Graph Stat. 2009;18(2):349-367.
                doi:10.1198/jcgs.2009.06134.
            [2] Roberts GO, Rosenthal JS. Coupling and ergodicity of adaptive Markov chain Monte Carlo algorithms.
                J Appl Probab. 2007;44(March 2005):458-475. doi:10.1239/jap/1183667414.
        """
        super().__init__(ll_func, log_prior_func, x0, proposal_stds, **kwargs)
        self._batch_size = batch_size
        self._min_val = min_val
        self._max_val = max_val
        self._acceptance_counter = np.zeros((self._nmr_problems, self._nmr_params), dtype=np.uint64, order='C')

    def _get_mcmc_method_kernel_data_elements(self):
        kernel_data = super()._get_mcmc_method_kernel_data_elements()
        kernel_data.update({
            'acceptance_counter': Array(self._acceptance_counter, mode='rw', ensure_zero_copy=True)
        })
        return kernel_data

    def _at_acceptance_callback_c_func(self):
        return '''
            void _sampleAccepted(_mcmc_method_data* method_data, ulong current_iteration, uint parameter_ind){
                method_data->acceptance_counter[parameter_ind]++;
            }
        '''

    def _get_proposal_update_function(self, nmr_samples, thinning, return_output):
        kernel_source = '''
            void _updateProposalState(_mcmc_method_data* method_data, ulong current_iteration,
                                      global mot_float_type* current_position){    

                if(current_iteration > 0 && current_iteration % ''' + str(self._batch_size) + ''' == 0){
                    for(uint k = 0; k < ''' + str(self._nmr_params) + '''; k++){
                        method_data->proposal_stds[k] *= sqrt(
                            ((mot_float_type)method_data->acceptance_counter[k] + 1) /
                            (''' + str(self._batch_size) + ''' - method_data->acceptance_counter[k] + 1));
                        
                        method_data->proposal_stds[k] = clamp(method_data->proposal_stds[k], 
                                                        (mot_float_type)''' + str(self._min_val) + ''', 
                                                        (mot_float_type)''' + str(self._max_val) + ''');

                        method_data->acceptance_counter[k] = 0;
                    }
                }             
            }
        '''
        return kernel_source
