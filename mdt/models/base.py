__author__ = 'Robbert Harms'
__date__ = "2015-10-27"
__maintainer__ = "Robbert Harms"
__email__ = "robbert@xkls.nl"


class EstimableModel:

    def __init__(self, *args, **kwargs):
        """This is an interface for all methods needed to be able to optimize and sample a model.
        """
        super().__init__()

    @property
    def name(self):
        """The name of this model.

        Returns:
            str: the name of this model.
        """
        raise NotImplementedError()

    def set_input_data(self, input_data, suppress_warnings=False):
        """Set the input data this model will deal with.

        Args:
            input_data (mdt.lib.input_data.MRIInputData): The container for the data we will use for this model.
            suppress_warnings (boolean): set to suppress all warnings

        Returns:
            Returns self for chainability
        """
        raise NotImplementedError()

    def get_used_volumes(self, input_data=None):
        """Get the indices of the input data volumes used by this model.

        Args:
            input_data (mdt.lib.input_data.MRIInputData): if given, limit the analysis to this input data.

        Returns:
            List: the list of volume indices with volumes to use. This indexes the input data.
        """
        raise NotImplementedError()

    def get_nmr_observations(self):
        """Get the number of observations in the data.

        Returns:
            int: the number of observations present in the data
        """
        raise NotImplementedError()

    def get_nmr_parameters(self):
        """Get the number of estimable parameters in this model.

        Returns:
            int: the number of estimable parameters
        """
        raise NotImplementedError()

    def get_lower_bounds(self):
        """Get the lower bounds.

        Returns:
            List: for every parameter a lower bound which can either be None, a scalar or a vector with a
                lower bound per problem instance.
        """
        raise NotImplementedError()

    def get_upper_bounds(self):
        """Get the upper bounds.

        Returns:
            List: for every parameter an upper bound which can either be None, a scalar or a vector with a
                upper bound per problem instance.
        """
        raise NotImplementedError()

    def get_free_param_names(self):
        """Get the names of the free parameters.

        Returns:
            List: the name of the free parameters
        """
        raise NotImplementedError()

    def get_kernel_data(self):
        """Get the kernel data this model needs for evaluation in OpenCL.

        This is needed for evaluating the priors, likelihoods and other functions.

        Returns:
            mot.lib.kernel_data.KernelData: the kernel data used by this model
        """
        raise NotImplementedError()

    def get_objective_function(self):
        """For minimization, get the negative of the log-likelihood function.

        Returns:
            mot.lib.cl_function.CLFunction: the CL function for the optimization routines in MOT.
        """
        raise NotImplementedError()

    def get_constraints_function(self):
        """The function for the (inequality) constraints.

        Returns:
            mot.lib.cl_function.CLFunction: the CL function for the inequality constraints of this model.
        """
        raise NotImplementedError()

    def get_log_likelihood_function(self):
        """For sampling, get the log-likelihood function.

        Returns:
            mot.lib.cl_function.CLFunction: the CL function for the log likelihood function during MCMC sampling
        """
        raise NotImplementedError()

    def get_log_prior_function(self):
        """Get the prior function used during sampling.

        Returns:
            mot.lib.cl_function.CLFunction: the CL function for the log prior function during MCMC sampling
        """
        raise NotImplementedError()

    def get_finalize_proposal_function(self):
        """Get the function used to finalize the proposal.

        This function is used to finalize the proposals during sampling.

        Returns:
            mot.lib.cl_function.CLFunction: the CL function used to finalize a proposal during sampling.
        """
        raise NotImplementedError()

    def get_initial_parameters(self):
        """Get the initial parameters for each of the voxels.

        Returns:
            ndarray: 2d array with for every problem (first dimension) the initial parameters (second dimension).
        """
        raise NotImplementedError()

    def get_post_optimization_output(self, optimized_parameters, roi_indices=None, parameters_dict=None):
        """Get the output after optimization.

        This is called by the processing strategy to finalize the optimization of a batch of voxels.

        Args:
            optimized_parameters (ndarray): the array of optimized parameters
            roi_indices (Iterable or None): if set, the problem instances optimized in this batch
            parameters_dict (dict): same data as ``optimized_parameters``, only then represented as a dict.
                Only needed if available, to speed up this function.

        Returns:
            dict: dictionary with results maps, can be nested which should translate to sub-directories.
        """
        raise NotImplementedError()

    def get_rwm_proposal_stds(self):
        """Get the Random Walk Metropolis proposal standard deviations for every parameter and every problem instance.

        These proposal standard deviations are used in Random Walk Metropolis MCMC sample.

        Returns:
            ndarray: the proposal standard deviations of each free parameter, for each problem instance
        """
        raise NotImplementedError()

    def get_rwm_epsilons(self):
        """Get per parameter a value small relative to the parameter's standard deviation.

        This is used in, for example, the SCAM Random Walk Metropolis sampling routine to add to the new proposal
        standard deviation to ensure it does not collapse to zero.

        Returns:
            list: per parameter an epsilon, relative to the proposal standard deviation
        """
        raise NotImplementedError()

    def get_random_parameter_positions(self, nmr_positions=1):
        """Get one or more random parameter positions centered around the initial parameters.

        This can be used to generate random starting points for sampling routines with multiple walkers.

        Per position requested, this function generates a normal distribution around the initial parameters (using
        :meth:`get_initial_parameters`) with the standard deviation derived from the random walk metropolis std.

        Returns:
            ndarray: a 3d matrix for (voxels, parameters, nmr_positions).
        """
        raise NotImplementedError()

    def get_post_sampling_maps(self, sampling_output, roi_indices=None):
        """Get the post sampling volume maps.

        This will return a dictionary mapping folder names to dictionaries with volumes to write.

        Args:
            sampling_output (mot.sample.base.SamplingOutput): the output of the sampler
            roi_indices (Iterable or None): if set, the problem instances sampled in this batch

        Returns:
            dict: a dictionary with for every subdirectory the maps to save
        """
        raise NotImplementedError()

    def get_mle_codec(self):
        """Get a parameter codec that can be used to transform the parameters to and from optimization and model space.

        Returns:
            mdt.model_building.utils.ParameterCodec: an instance of a parameter codec
        """
        raise NotImplementedError()

    def update_active_post_processing(self, processing_type, settings):
        """Update the active post-processing semaphores.

        It is possible to control which post-processing routines get run by overwriting them using this method.
        For a list of post-processors, please see the default mdt configuration file under ``active_post_processing``.

        Args:
            processing_type (str): one of ``sample`` or ``optimization``.
            settings (dict): the items to set in the post-processing information
        """
        raise NotImplementedError()

    def is_input_data_sufficient(self, input_data=None):
        """Check if the input data has enough information for this model to work.

        Args:
            input_data (mdt.lib.input_data.MRIInputData): The input data we intend on using with this model.

        Returns:
            boolean: True if there is enough information in the input data, false otherwise.
        """
        raise NotImplementedError()

    def get_input_data_problems(self, input_data=None):
        """Get all the problems with the protocol.

        Args:
            input_data (mdt.lib.input_data.MRIInputData): The input data we intend on using with this model.

        Returns:
            list of InputDataProblem: A list of
                InputDataProblem instances or subclasses of that baseclass.
                These objects indicate the problems with the protocol and this model.
        """
        raise NotImplementedError()


class InputDataProblem:

    def __init__(self):
        """The base class for indicating problems with the input data.

        These are meant to be returned from the function get_input_data_problems().

        Each of these problems is supposed to overwrite the function __str__() for reporting the problem.
        """

    def __repr__(self):
        return self.__str__()


class MissingProtocolInput(InputDataProblem):

    def __init__(self, missing_columns):
        super().__init__()
        self.missing_columns = missing_columns

    def __str__(self):
        return 'Missing columns: ' + ', '.join(self.missing_columns)


class NamedProtocolProblem(InputDataProblem):

    def __init__(self, model_protocol_problem, model_name):
        """This extends the given model protocol problem to also include the name of the model.

        Args:
            model_protocol_problem (InputDataProblem): The name for the problem with the given model.
            model_name (str): the name of the model
        """
        super().__init__()
        self._model_protocol_problem = model_protocol_problem
        self._model_name = model_name

    def __str__(self):
        return "{0}: {1}".format(self._model_name, self._model_protocol_problem)
