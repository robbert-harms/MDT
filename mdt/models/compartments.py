from mdt.model_building.model_functions import SimpleModelCLFunction

__author__ = 'Robbert Harms'
__date__ = "2015-12-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRICompartmentModelFunction(SimpleModelCLFunction):

    def __init__(self, name, cl_function_name, parameters, cl_body, dependencies, return_type,
                 model_function_priors=None, post_optimization_modifiers=None, extra_optimization_maps_funcs=None,
                 extra_sampling_maps_funcs=None,
                 cl_extra=None, proposal_callbacks=None):
        """Create a new dMRI compartment model function.

        Args:
            name (str): the name of this compartment model
            cl_function_name (str): the name of this function in the CL kernel
            parameters (list of CLFunctionParameter): the list of the function parameters
            cl_body (str): the body of the CL code
            dependencies (list): the list of functions we depend on inside the kernel
            return_type (str): the CL return type
            model_function_priors (list of mot.lib.cl_function.CLFunction): additional
                compartment priors on top of the parameter priors.
            post_optimization_modifiers (None or list or tuple): a list of modification callbacks to alter the
                optimized point. These modifiers are supposed to be called before the post optimization modifiers
                of the composite model.
            extra_optimization_maps_funcs (None or list or tuple): a list of modification callbacks to add new maps
                after optimization.
            extra_sampling_maps_funcs (None or list or tuple): a list of functions that can return additional maps
                after sample.
            cl_extra (str): optional extra CL code outside of the function body
            proposal_callbacks (List[Tuple(Tuple(CLFunctionParameter), mot.lib.cl_function.CLFunction)]): additional
                proposal callback functions. These are (indirectly) called by the MCMC sampler to finalize every
                proposal.
        """
        super().__init__(return_type, name, cl_function_name,
                                                           parameters, cl_body,
                                                           dependencies=dependencies,
                                                           model_function_priors=model_function_priors,
                                                           cl_extra=cl_extra)
        self.post_optimization_modifiers = post_optimization_modifiers or []
        self.extra_optimization_maps_funcs = extra_optimization_maps_funcs or []
        self.extra_sampling_maps_funcs = extra_sampling_maps_funcs or []
        self.proposal_callbacks = proposal_callbacks or []
