from mdt.model_building.computation_caching import DataCacheParameter, CacheStruct
from mdt.model_building.model_functions import SimpleModelCLFunction, WeightType, ModelCLFunction
from mot.lib.cl_function import SimpleCLFunction

__author__ = 'Robbert Harms'
__date__ = "2015-12-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CompartmentModel(ModelCLFunction):

    def get_post_optimization_modifiers(self):
        """Get a list of post optimization modification callbacks.

        These functions are the first to be called after model optimization, allowing the compartment to transform
        the optimization results.

        Returns:
             list of functions: the list of functions to be called
        """
        raise NotImplementedError()

    def get_extra_optimization_maps_funcs(self):
        """Get a list of functions to produce additional results post optimization.

        These functions are called after all post optimization modifiers. In contrast to the modifiers, these
        functions can return additional maps as optimization output.

        Returns:
            list of functions: the list of functions to be called
        """
        raise NotImplementedError()

    def get_extra_sampling_maps_funcs(self):
        """Get a list of functions to produce additional results post sampling.

        These functions can return additional maps as sampling output.

        Returns:
            list of functions: the list of functions to be called
        """
        raise NotImplementedError()

    def get_proposal_callbacks(self):
        """Get a list of proposal callbacks.

        These are (indirectly) called by the MCMC sampler to finalize every proposal. That is, they can change the
        proposal before it is handed to the prior and likelihood function.

        Returns:
            List[Tuple(Tuple(CLFunctionParameter), mot.lib.cl_function.CLFunction)]: a list with all the
                information per callback
        """
        raise NotImplementedError()

    def get_cache_info(self):
        """Get the cache info for this compartment model.

        Returns:
            None or mdt.model_building.computation_caching.CacheInfo: the cache info, or None if no caching is needed
        """
        raise NotImplementedError()


class DMRICompartmentModelFunction(CompartmentModel, SimpleModelCLFunction):

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
            cl_extra (str): optional extra CL code outside of the function body.
            model_function_priors (list of mot.lib.cl_function.CLFunction): additional
                compartment priors on top of the parameter priors.
            post_optimization_modifiers (None or list or tuple): a list of modification callbacks to alter the
                optimized point.
            extra_optimization_maps_funcs (None or list or tuple): a list of modification callbacks to add new maps
                after optimization.
            extra_sampling_maps_funcs (None or list or tuple): a list of functions that can return additional maps
                after sampling.
            proposal_callbacks (List[Tuple(Tuple(CLFunctionParameter), mot.lib.cl_function.CLFunction)]): additional
                proposal callback functions. These are (indirectly) called by the MCMC sampler to finalize every
                proposal.
            nickname (str or None): the nickname of this compartment model function. If given, this is the name of this
                compartment in a composite model function tree
            cache_info (mdt.model_building.computation_caching.CacheInfo): the cache information for this compartment
        """
        super().__init__(return_type, cl_function_name, parameters, cl_body, dependencies=dependencies,
                         model_function_priors=model_function_priors, cl_extra=cl_extra)
        self._nickname = nickname
        self._post_optimization_modifiers = post_optimization_modifiers or []
        self._extra_optimization_maps_funcs = extra_optimization_maps_funcs or []
        self._extra_sampling_maps_funcs = extra_sampling_maps_funcs or []
        self._proposal_callbacks = proposal_callbacks or []
        self._cache_info = cache_info

    @property
    def name(self):
        return self._nickname or self.get_cl_function_name()

    def get_post_optimization_modifiers(self):
        return self._post_optimization_modifiers

    def get_extra_optimization_maps_funcs(self):
        return self._extra_optimization_maps_funcs

    def get_extra_sampling_maps_funcs(self):
        return self._extra_sampling_maps_funcs

    def get_proposal_callbacks(self):
        return self._proposal_callbacks

    def get_cache_info(self):
        return self._cache_info

    def get_cache_struct(self, address_space):
        """Get the CL code for the cache struct of this compartment.

        Args:
            address_space (str): the CL address space for the primitive elements of this struct

        Returns:
            str: the CL code for the cache struct
        """
        return self._get_cache_struct().get_type_definitions(address_space)

    def get_cache_init_function(self, address_space):
        """Get the CL function for initializing the cache struct of this compartment.

        Returns:
            mot.lib.cl_function.CLFunction: the CL function for initializing the cache. This has the same
                signature as the compartment model function.
        """
        dependency_calls = []
        cache_init_funcs = []
        for dependency in self.get_dependencies():
            if isinstance(dependency, CompartmentModel):
                cache_init_func = dependency.get_cache_init_function(address_space)
                params = []
                for p in cache_init_func.get_parameters():
                    if isinstance(p, DataCacheParameter):
                        params.append('{}->{}'.format(p.name, dependency.get_cl_function_name()))
                    else:
                        params.append(p.name)
                dependency_calls.append('{}({});'.format(cache_init_func.get_cl_function_name(), ', '.join(params)))
                cache_init_funcs.append(cache_init_func)

        return SimpleCLFunction(
            'void',
            '{}_init_cache'.format(self._function_name),
            self._parameter_list,
            self._cache_info.cl_code + '\n'.join(dependency_calls),
            dependencies=cache_init_funcs + self.get_dependencies(),
            cl_extra=self._get_cache_struct().get_type_definitions(address_space))

    def evaluate(self, *args, **kwargs):
        if not any(isinstance(p, DataCacheParameter) for p in self._parameter_list):
            return super().evaluate(*args, **kwargs)

        cache_struct = self._get_cache_struct()

        # def _get_cache_func():
        #     for dependency in self.get_dependencies():
        #         if isinstance(dependency, DataCacheFunction):
        #             return dependency
        #
        def _get_call_parameters():
            params = []
            for p in self._parameter_list:
                if isinstance(p, DataCacheParameter):
                    params.append(cache_struct.get_struct_initialization(self.name))
                else:
                    params.append(p.name)
            return params


        cache_init_func = self.get_cache_init_function('private')

        with_cache_func = SimpleCLFunction(
            self._return_type,
            '_{}'.format(self._function_name),
            [p for p in self._parameter_list if not isinstance(p, DataCacheParameter)],
            '''
                {initialize_struct}
                {cache_init_func_name}({params});
                return {func_name}({params});
                
            '''.format(#cache_func_name=_get_cache_func().get_cl_function_name(),

                        initialize_struct=cache_struct.initialize_variable(self.name, 'private'),
                cache_init_func_name=cache_init_func.get_cl_function_name(),

                       func_name=self.get_cl_function_name(),
                       params=', '.join(_get_call_parameters())),
            dependencies=[cache_init_func, self],
            # cl_extra=(self.get_cl_extra() or '')# + self.get_cache_struct('private')
            #           + self.get_cl_code()# + cache_init_func.get_cl_code()

        #      //{cache_func_name}({params});
            #                 //return {func_name}({params});
        )

        print(with_cache_func.get_cl_code())
        return with_cache_func.evaluate(*args, **kwargs)

    def _get_cache_struct(self):
        dependencies = []
        for dependency in self.get_dependencies():
            if isinstance(dependency, CompartmentModel):
                dependencies.append(CacheStruct(dependency.get_cache_info().fields, dependency.get_cl_function_name()))

        fields = list(self._cache_info.fields) + dependencies
        return CacheStruct(fields, self.get_cl_function_name())


class WeightCompartment(CompartmentModel, WeightType):

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
        self._nickname = nickname

    @property
    def name(self):
        return self._nickname or self.get_cl_function_name()

    def get_post_optimization_modifiers(self):
        return []

    def get_extra_optimization_maps_funcs(self):
        return []

    def get_extra_sampling_maps_funcs(self):
        return []

    def get_proposal_callbacks(self):
        return []

    def get_cache_info(self):
        return None
