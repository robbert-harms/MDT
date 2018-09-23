from mdt.model_building.computation_caching import DataCacheParameter, CacheStruct
from mdt.model_building.model_functions import SimpleModelCLFunction, WeightType
from mot.lib.cl_function import SimpleCLFunction

__author__ = 'Robbert Harms'
__date__ = "2015-12-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DMRICompartmentModelFunction(SimpleModelCLFunction):

    def __init__(self, return_type, cl_function_name, parameters, cl_body, dependencies=None,
                 cl_extra=None, model_function_priors=None, post_optimization_modifiers=None,
                 extra_optimization_maps_funcs=None, extra_sampling_maps_funcs=None, proposal_callbacks=None,
                 nickname=None, cache_info=None):
        """Create a new dMRI compartment model function.

        Args:
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
            nickname (str or None): the nickname of this compartment model function. If given, this is the name of this
                compartment in a composite model function tree
            cache_info (mdt.model_building.computation_caching.CacheInfo): the cache information for this compartment
        """
        super().__init__(return_type, cl_function_name, parameters, cl_body, dependencies=dependencies,
                         model_function_priors=model_function_priors, cl_extra=cl_extra)
        self.post_optimization_modifiers = post_optimization_modifiers or []
        self.extra_optimization_maps_funcs = extra_optimization_maps_funcs or []
        self.extra_sampling_maps_funcs = extra_sampling_maps_funcs or []
        self.proposal_callbacks = proposal_callbacks or []
        self.nickname = nickname
        self.cache_info = cache_info

    @property
    def name(self):
        """Get the name of this model function, for use in composite model functions

        Returns:
            str: Either the nickname or the CL function name
        """
        return self.nickname or self.get_cl_function_name()

    def get_cache_struct(self, address_space):
        """Get the CL code for the cache struct of this compartment.

        Args:
            address_space (str): the CL address space for the primitive elements of this struct

        Returns:
            str: the CL code for the cache struct
        """
        return self._get_cache_struct().get_type_definitions(address_space)

    def get_cache_init_function(self):
        """Get the CL function for initializing the cache struct of this compartment.

        Returns:
            mot.lib.cl_function.CLFunction: the CL function for initializing the cache. This has the same
                signature as the compartment model function.
        """
        dependency_calls = []
        for dependency in self.get_dependencies():
            if isinstance(dependency, DMRICompartmentModelFunction):
                params = []
                for p in self._parameter_list:
                    if isinstance(p, DataCacheParameter):
                        params.append('{}->{}'.format(p.name, dependency.get_cl_function_name()))
                    else:
                        params.append(p.name)
                dependency_calls.append('{}({});'.format(dependency.get_cl_function_name(), ', '.join(params)))

        return SimpleCLFunction(
            self._return_type,
            '{}_init_cache'.format(self._function_name),
            self._parameter_list,
            self.cache_info.cl_code + '\n'.join(dependency_calls))

    def evaluate(self, *args, **kwargs):
        if not any(isinstance(p, DataCacheParameter) for p in self._parameter_list):
            return super().evaluate(*args, **kwargs)

        non_cache_params = [p for p in self._parameter_list if not isinstance(p, DataCacheParameter)]

        # def _get_cache_func():
        #     for dependency in self.get_dependencies():
        #         if isinstance(dependency, DataCacheFunction):
        #             return dependency
        #
        # def _get_cache_parameter_name():
        #     for p in self._parameter_list:
        #         if isinstance(p, DataCacheParameter):
        #             return p.name


        #
        # cache_init_func = self.get_cache_init_function()
        #
        # with_cache_func = SimpleCLFunction(
        #     self._return_type,
        #     '_{}'.format(self._function_name),
        #     non_cache_params,
        #     '',
        #     # '''
        #     #     {cache_func_name}({params});
        #     #     return {func_name}({params});
        #     # '''.format(cache_func_name=_get_cache_func().get_cl_function_name(),
        #     #            func_name=self.get_cl_function_name(),
        #     #            params=', '.join(p.name for p in self.get_parameters())),
        #     cl_extra=(self.get_cl_extra() or '') + self.get_cache_struct('private')
        #               + self.get_cl_code() + cache_init_func.get_cl_code()
        # )
        #
        # print(with_cache_func.get_cl_code())
        #
        # with_cache_func.evaluate(*args, **kwargs)
        #
        # if isinstance(inputs, Iterable) and not isinstance(inputs, Mapping):
        #     inputs = list(inputs)
        #     param_names = [p.name for p in self.get_parameters() if not isinstance(p, DataCacheParameter)]
        #     inputs = dict(zip(param_names, inputs))
        #
        # inputs.update({_get_cache_parameter_name(): _get_cache_func().data_struct})
        #
        # return with_cache_func.evaluate(inputs, nmr_instances, use_local_reduction=use_local_reduction,
        #                                 cl_runtime_info=cl_runtime_info)

    def _get_cache_struct(self):
        dependencies = []
        for dependency in self.get_dependencies():
            if isinstance(dependency, DMRICompartmentModelFunction):
                dependencies.append(CacheStruct(dependency.cache_info.fields, dependency.get_cl_function_name()))

        fields = list(self.cache_info.fields) + dependencies
        return CacheStruct(fields, self.get_cl_function_name())


class WeightCompartment(WeightType):

    def __init__(self, return_type, cl_function_name, parameters, cl_body, dependencies=None,
                 cl_extra=None, model_function_priors=None, nickname=None):
        """Create a new weight for use in composite models

        Args:
            cl_function_name (str): the name of this function in the CL kernel
            parameters (list of CLFunctionParameter): the list of the function parameters
            cl_body (str): the body of the CL code
            dependencies (list): the list of functions we depend on inside the kernel
            return_type (str): the CL return type
            model_function_priors (list of mot.lib.cl_function.CLFunction): additional
                compartment priors on top of the parameter priors.
            nickname (str or None): the nickname of this compartment model function. If given, this is the name of this
                compartment in a composite model function tree
        """
        super().__init__(return_type, cl_function_name, parameters, cl_body, dependencies=dependencies,
                         model_function_priors=model_function_priors, cl_extra=cl_extra)
        self.nickname = nickname

    @property
    def name(self):
        """Get the name of this model function, for use in composite model functions

        Returns:
            str: Either the nickname or the CL function name
        """
        return self.nickname or self.get_cl_function_name()
