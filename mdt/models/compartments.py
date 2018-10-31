from mdt.model_building.model_functions import SimpleModelCLFunction, WeightType, ModelCLFunction
from mdt.model_building.parameters import FreeParameter, DataCacheParameter, NoiseStdInputParameter
from mot.lib.cl_function import SimpleCLFunction
from mot.lib.kernel_data import Struct, PrivateMemory, LocalMemory

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

    def get_cache_struct(self, address_space):
        """Get the cache data for this compartment model.

        Args:
            address_space (str): the address space for the cache, either 'local' or 'private'

        Returns:
            Optional[mdt.model_building.computation_caching.CacheStruct]: the cache structure
        """
        raise NotImplementedError()

    def get_cache_init_function(self):
        """Get the CL function for initializing the cache struct of this compartment.

        Please note that this function does not the struct type definitions of the cache. These need to be
        appended manually using the :meth:`get_cache_struct`.

        Returns:
            None or mot.lib.cl_function.CLFunction: the CL function for initializing the cache. This function
                should have the same signature as the compartment model function.
        """
        raise NotImplementedError()


class DMRICompartmentModelFunction(CompartmentModel, SimpleModelCLFunction):

    def __init__(self, return_type, cl_function_name, parameters, cl_body, dependencies=None,
                 model_function_priors=None, post_optimization_modifiers=None,
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
            cache_info (Optional[CacheInfo]): the cache information for this compartment
        """
        super().__init__(return_type, cl_function_name, parameters, cl_body, dependencies=dependencies,
                         model_function_priors=model_function_priors)
        self._nickname = nickname
        self._post_optimization_modifiers = post_optimization_modifiers or []
        self._extra_optimization_maps_funcs = extra_optimization_maps_funcs or []
        self._extra_sampling_maps_funcs = extra_sampling_maps_funcs or []
        self._proposal_callbacks = proposal_callbacks or []
        self._cache_info = cache_info

        if not self._cache_info and len([p for p in parameters if isinstance(p, DataCacheParameter)]):
            self._cache_info = CacheInfo([], '')

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

    def get_cache_struct(self, address_space):
        if not self._cache_info:
            return None

        fields = {}
        for dependency in self.get_dependencies():
            if isinstance(dependency, CompartmentModel):
                if dependency.get_cache_struct(address_space):
                    fields.update(dependency.get_cache_struct(address_space))

        struct_name = self._get_cache_parameter().ctype

        if address_space == 'private':
            for ctype, name, nmr_elements in self._cache_info.fields:
                fields[name] = PrivateMemory(nmr_elements, ctype)
        else:
            for ctype, name, nmr_elements in self._cache_info.fields:
                fields[name] = LocalMemory(ctype, nmr_items=nmr_elements)

        return {self.name: Struct(fields, struct_name)}

    def get_cache_init_function(self):
        if not self.get_cache_struct('private'):
            return None

        dependency_calls = []
        cache_init_funcs = []
        for dependency in self.get_dependencies():
            if isinstance(dependency, CompartmentModel):
                cache_init_func = dependency.get_cache_init_function()
                if cache_init_func:
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
            [p for p in self.get_parameters()
             if isinstance(p, (FreeParameter, DataCacheParameter, NoiseStdInputParameter))],
            self._cache_info.cl_code + '\n'.join(dependency_calls),
            dependencies=cache_init_funcs + self.get_dependencies())

    def evaluate(self, *args, **kwargs):
        if not any(isinstance(p, DataCacheParameter) for p in self._parameter_list):
            return super().evaluate(*args, **kwargs)

        cache_struct = self.get_cache_struct('private')[self.name]
        cache_param_name = self._get_cache_parameter().name

        if isinstance(args[0], (tuple, list)):
            args = list(args)
            args[0] = tuple(args[0]) + (cache_struct,)
        else:
            args[0][cache_param_name] = cache_struct

        cache_init_func = self.get_cache_init_function()

        with_cache_func = SimpleCLFunction(
            self._return_type,
            '_{}'.format(self._function_name),
            self._parameter_list,
            '''
                {cache_init_func_name}({cache_params});
                return {parent_func_name}({parent_func_params});
            '''.format(cache_init_func_name=cache_init_func.get_cl_function_name(),
                       parent_func_name=self.get_cl_function_name(),
                       cache_params=', '.join([p.name for p in cache_init_func.get_parameters()]),
                       parent_func_params=', '.join([p.name for p in self.get_parameters()])),
            dependencies=[cache_init_func, self])

        return with_cache_func.evaluate(*args, **kwargs)

    def _get_cache_parameter(self):
        for p in self.get_parameters():
            if isinstance(p, DataCacheParameter):
                return p


class WeightCompartment(CompartmentModel, WeightType):

    def __init__(self, return_type, cl_function_name, parameters, cl_body, dependencies=None,
                 model_function_priors=None, nickname=None):
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
                         model_function_priors=model_function_priors)
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

    def get_cache_struct(self, address_space):
        return None

    def get_cache_init_function(self):
        return None


class CacheInfo:

    def __init__(self, fields, cl_code):
        """A storage container for the cache info of an compartment function.

        Args:
            fields (list of tuple): the list of fields used in the cache. Each tuple element consists of
                (ctype, name, nmr_elements).
            cl_code (str): the body of the CL code needed for the cache initialization.
                This will be turned into a function by the compartment model.
        """
        self.fields = fields
        self.cl_code = cl_code
