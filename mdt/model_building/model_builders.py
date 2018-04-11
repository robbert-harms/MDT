import inspect
from textwrap import dedent, indent

import numpy as np
import copy
from six import string_types
from mot.cl_data_type import SimpleCLDataType
from mot.cl_function import SimpleCLFunction
from mot.cl_function import SimpleCLFunctionParameter
from mot.cl_routines.mapping.codec_runner import CodecRunner
from .model_functions import WeightType, ModelCLFunction
from .parameters import CurrentObservationParam, StaticMapParameter, ProtocolParameter, FreeParameter
from .parameter_functions.dependencies import SimpleAssignment, AbstractParameterDependency
from .utils import ParameterCodec
from mot.model_interfaces import OptimizeModelInterface, SampleModelInterface, NumericalDerivativeInterface
from mot.utils import is_scalar, all_elements_equal, get_single_value, NameFunctionTuple, convert_data_to_dtype, \
    get_class_that_defined_method
from mot.kernel_input_data import KernelInputArray

__author__ = 'Robbert Harms'
__date__ = "2014-03-14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ModelBuilder(object):
    """The interface for a model builder.

    The primary method of a model builder is the ``build`` method which can be used to build a
    ``OptimizeModelInterface`` or subclasses thereof.
    """

    def build(self, problems_to_analyze=None):
        """Construct the final model using all current construction settings.

        Args:
            problems_to_analyze (ndarray): optional set of problem indices, this should construct the model
                such that it analyzes only the indicated subset of problems.

        Returns:
            OptimizeModelInterface or SampleModelInterface: an implementation of an optimization or
                optimization/sampling model using the current construction settings.
        """
        raise NotImplementedError()


class OptimizeModelBuilder(ModelBuilder):

    def __init__(self, name, model_tree, likelihood_function, signal_noise_model=None,
                 input_data=None, enforce_weights_sum_to_one=True):
        """Create a new model builder that can construct an optimization model from a combination of model functions.

        Args:
            name (str): the name of the model
            model_tree (mdt.model_building.trees.CompartmentModelTree): the model tree object
            likelihood_function (mdt.model_building.likelihood_functions.LikelihoodFunction): the likelihood function to
                use for the resulting complete model.
            signal_noise_model (mdt.model_building.signal_noise_models.SignalNoiseModel): the optional signal
                noise model to use to add noise to the model prediction
            input_data (mdt.model_building.input_data.InputData): the input data container
            enforce_weights_sum_to_one (boolean): if we want to enforce that weights sum to one. This does the
                following things; it fixes the first weight to the sum of the others and it adds a transformation
                that ensures that those other weights sum to at most one.
        """
        super(OptimizeModelBuilder, self).__init__()
        self._name = name
        self._model_tree = model_tree
        self._likelihood_function = likelihood_function
        self._signal_noise_model = signal_noise_model

        self._enforce_weights_sum_to_one = enforce_weights_sum_to_one

        self._model_functions_info = self._init_model_information_container(
            model_tree, likelihood_function, signal_noise_model)

        self._lower_bounds = {'{}.{}'.format(m.name, p.name): p.lower_bound for m, p in
                              self._model_functions_info.get_free_parameters_list()}

        self._upper_bounds = {'{}.{}'.format(m.name, p.name): p.upper_bound for m, p in
                              self._model_functions_info.get_free_parameters_list()}

        self._input_data = None
        if input_data:
            self.set_input_data(input_data)

        self._set_default_dependencies()

    @property
    def name(self):
        return self._name

    def _init_model_information_container(self, model_tree, likelihood_function, signal_noise_model):
        """Get the model information container object.

        The rationale for this function is that some subclasses may have additional parameters not present in
        optimization. For example, in sampling one can have priors with parameters. These parameters must be
        added to the model and the best point to do that is in the ModelFunctionsInformation object.

        Returns:
            ModelFunctionsInformation: the model function information object
        """
        return ModelFunctionsInformation(model_tree, likelihood_function, signal_noise_model)

    def get_composite_model_function(self):
        """Get the composite model function for the current model tree.

        The output model function of this class is a subclass of :class:`~mot.cl_function.CLFunction` meaning it can
        be used to evaluate the model given some input parameters.

        This function does not incorporate the likelihood function (Gaussian, Rician, etc.), but does incorporate the
        signal noise model (JohnsonNoise for example).

        Returns:
            CompositeModelFunction: the model function for the composite model
        """
        return CompositeModelFunction(self._model_tree, signal_noise_model=self._signal_noise_model)

    def build(self, problems_to_analyze=None):
        """Construct the final immutable model with the current settings.

        Args:
            problems_to_analyze (ndarray): construct the model such that it analyzes only a subset of the problems

        Returns:
            OptimizeModelInterface: an implementation an optimization model with all the current settings

        Raises:
            RuntimeError: if some of the required items are not set prior to building.
        """
        if self._input_data is None:
            raise RuntimeError('Input data is not set, can not build the model.')

        return SimpleOptimizeModel(problems_to_analyze,
                                   self.name,
                                   self._get_kernel_data(problems_to_analyze),
                                   self._get_nmr_problems(problems_to_analyze),
                                   self.get_nmr_inst_per_problem(),
                                   self.get_nmr_parameters(),
                                   self._get_initial_parameters(problems_to_analyze),
                                   self._get_pre_eval_parameter_modifier(),
                                   self._get_objective_per_observation_function(problems_to_analyze),
                                   self.get_lower_bounds(),
                                   self.get_upper_bounds(),
                                   self._get_max_numdiff_step(),
                                   self._get_numdiff_scaling_factors(),
                                   self._get_numdiff_use_bounds(),
                                   self._get_numdiff_use_lower_bounds(),
                                   self._get_numdiff_use_upper_bounds(),
                                   self._get_numdiff_param_transform())

    def get_parameter_codec(self):
        """Get a parameter codec that can be used to transform the parameters to and from optimization and model space.

        This is typically used as input to the ParameterTransformedModel decorator model.

        Returns:
            mdt.model_building.utils.ParameterCodec: an instance of a parameter codec
        """
        model_builder = self

        class Codec(ParameterCodec):
            def get_parameter_decode_function(self, function_name='decodeParameters'):
                func = '''
                    void ''' + function_name + '''(mot_data_struct* data, mot_float_type* x){
                '''
                for d in model_builder._get_parameter_transformations()[1]:
                    func += "\n" + "\t" * 4 + d.format('x')

                if model_builder._enforce_weights_sum_to_one:
                    func += model_builder._get_weight_sum_to_one_transformation()

                return func + '''
                    }
                '''

            def get_parameter_encode_function(self, function_name='encodeParameters'):
                func = '''
                    void ''' + function_name + '''(mot_data_struct* data, mot_float_type* x){
                '''

                if model_builder._enforce_weights_sum_to_one:
                    func += model_builder._get_weight_sum_to_one_transformation()

                for d in model_builder._get_parameter_transformations()[0]:
                    func += "\n" + "\t" * 4 + d.format('x')

                return func + '''
                    }
                '''
        return Codec()

    def fix(self, model_param_name, value):
        """Fix the given model.param to the given value.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'
            value (scalar or vector or string or AbstractParameterDependency): The value or dependency
                to fix the given parameter to.

        Returns:
            Returns self for chainability
        """
        if isinstance(value, string_types):
            value = SimpleAssignment(value)
        self._model_functions_info.fix_parameter(model_param_name, value)
        return self

    def unfix(self, model_param_name):
        """Unfix the given model.param

        Args:
            model_param_name (string): A model.param name like 'Ball.d'

        Returns:
            Returns self for chainability
        """
        self._model_functions_info.unfix(model_param_name)
        return self

    def init(self, model_param_name, value):
        """Init the given model.param to the given value.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'
            value (scalar or vector): The value to initialize the given parameter to

        Returns:
            Returns self for chainability
        """
        if not self._model_functions_info.is_fixed(model_param_name):
            self._model_functions_info.set_parameter_value(model_param_name, value)
        return self

    def set_initial_parameters(self, initial_params):
        """Update the initial parameters for this model by the given values.

        This only affects free parameters.

        Args:
            initial_params (dict): a dictionary containing as keys full parameter names (<model>.<param>) and as values
                numbers or arrays to be used as starting point
        """
        for m, p in self._model_functions_info.get_model_parameter_list():
            param_name = '{}.{}'.format(m.name, p.name)

            if param_name in initial_params:
                self.init(param_name, initial_params[param_name])

        return self

    def set_lower_bound(self, model_param_name, value):
        """Set the lower bound for the given parameter to the given value.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'
            value (scalar or vector): The value to set the lower bounds to

        Returns:
            Returns self for chainability
        """
        self._lower_bounds[model_param_name] = value
        return self

    def set_lower_bounds(self, lower_bounds):
        """Apply multiple lower bounds from a dictionary.

        Args:
            lower_bounds (dict): per parameter a lower bound

        Returns:
            Returns self for chainability
        """
        for param, value in lower_bounds.items():
            self.set_lower_bound(param, value)
        return self

    def set_upper_bound(self, model_param_name, value):
        """Set the upper bound for the given parameter to the given value.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'
            value (scalar or vector): The value to set the upper bounds to

        Returns:
            Returns self for chainability
        """
        self._upper_bounds[model_param_name] = value
        return self

    def set_upper_bounds(self, upper_bounds):
        """Apply multiple upper bounds from a dictionary.

        Args:
            upper_bounds (dict): per parameter a upper bound

        Returns:
            Returns self for chainability
        """
        for param, value in upper_bounds.items():
            self.set_upper_bound(param, value)
        return self

    def has_parameter(self, model_param_name):
        """Check to see if the given parameter is defined in this model.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'

        Returns:
            boolean: true if the parameter is defined in this model, false otherwise.
        """
        return self._model_functions_info.has_parameter(model_param_name)

    def set_input_data(self, input_data):
        """Set the input data this model will deal with.

        This will also call the function set_noise_level_std() with the noise_std from the new input data.

        Args:
            input_data (mdt.model_building.input_data.InputData):
                The container for the data we will use for this model.

        Returns:
            Returns self for chainability
        """
        self._input_data = input_data
        if self._input_data.noise_std is not None:
            self._model_functions_info.set_parameter_value('{}.{}'.format(
                self._likelihood_function.name,
                self._likelihood_function.get_noise_std_param_name()), self._input_data.noise_std)
        return self

    def get_input_data(self):
        """Get the input data actually being used by this model.

        Returns:
            mdt.model_building.input_data.InputData: the input data being used by this model
        """
        return self._input_data

    def set_fixed_parameter_values(self, fixed_values):
        """Given a dictionary with static maps, initialize the values of the static parameters with these values.

        Make sure that if vectors are given, the length of the vector should match the length of the number of problems
        (in the input data).

        Args:
            fixed_values (dict): the dictionary with the static maps.
        """
        static_params = self._model_functions_info.get_static_parameters_list()
        for m, p in static_params:
            if p.name in fixed_values:
                self._model_functions_info.set_parameter_value('{}.{}'.format(m.name, p.name), fixed_values[p.name])
        return self

    def get_nmr_inst_per_problem(self):
        """See super class for details"""
        return self._input_data.nmr_observations

    def get_nmr_parameters(self):
        """See super class for details"""
        return len(self._model_functions_info.get_estimable_parameters_list())

    def get_lower_bounds(self):
        """See super class for details"""
        return [self._lower_bounds['{}.{}'.format(m.name, p.name)] for m, p in
                self._model_functions_info.get_estimable_parameters_list()]

    def get_upper_bounds(self):
        """See super class for details"""
        return [self._upper_bounds['{}.{}'.format(m.name, p.name)] for m, p in
                self._model_functions_info.get_estimable_parameters_list()]

    def get_initial_parameters(self):
        starting_points = []
        for m, p in self._model_functions_info.get_estimable_parameters_list():
            param_name = '{}.{}'.format(m.name, p.name)
            starting_points.append(self._model_functions_info.get_parameter_value(param_name))
        return starting_points

    def _get_nmr_problems(self, problems_to_analyze):
        """See super class for details"""
        if problems_to_analyze is None:
            if self._input_data:
                return self._input_data.nmr_problems
            return 0
        return len(problems_to_analyze)

    def _get_kernel_data(self, problems_to_analyze):
        data_items = {}
        data_items.update(self._get_observations_data(problems_to_analyze))
        data_items.update(self._get_fixed_parameters_as_var_data(problems_to_analyze))
        data_items.update(self._get_static_parameters_as_var_data(problems_to_analyze))
        data_items.update(self._get_bounds_as_var_data(problems_to_analyze))
        data_items.update(self._get_protocol_data(problems_to_analyze))
        return data_items

    def _get_initial_parameters(self, problems_to_analyze):
        np_dtype = np.float32

        starting_points = []
        for m, p in self._model_functions_info.get_estimable_parameters_list():
            param_name = '{}.{}'.format(m.name, p.name)
            value = self._model_functions_info.get_parameter_value(param_name)

            if is_scalar(value):
                if self._get_nmr_problems(problems_to_analyze) == 0:
                    starting_points.append(np.full((1, 1), value, dtype=np_dtype))
                else:
                    starting_points.append(np.full((self._get_nmr_problems(problems_to_analyze), 1), value,
                                                   dtype=np_dtype))
            else:
                if len(value.shape) < 2:
                    value = np.transpose(np.asarray([value]))
                elif value.shape[1] > value.shape[0]:
                    value = np.transpose(value)
                else:
                    value = value

                if problems_to_analyze is None:
                    starting_points.append(value)
                else:
                    starting_points.append(value[problems_to_analyze, ...])

        starting_points = np.concatenate([np.transpose(np.array([s]))
                                          if len(s.shape) < 2 else s for s in starting_points], axis=1)

        return convert_data_to_dtype(starting_points, 'mot_float_type', SimpleCLDataType.from_string('float'))

    def _get_pre_eval_parameter_modifier(self):
        func_name = '_modifyParameters'
        func = '''
            void ''' + func_name + '''(mot_data_struct* data, mot_float_type* x){
            }
        '''
        return NameFunctionTuple(func_name, func)

    def _get_objective_per_observation_function(self, problems_to_analyze):
        eval_function_info = self._get_model_eval_function(problems_to_analyze)
        eval_model_func = self._likelihood_function.get_log_likelihood_function(include_constant_terms=False)

        eval_call_args = ['observation', 'model_evaluation']
        param_listing = ''
        for p in eval_model_func.get_free_parameters():
            param_listing += self._get_param_listing_for_param(eval_model_func, p)
            eval_call_args.append('{}.{}'.format(eval_model_func.name, p.name).replace('.', '_'))

        preliminary = ''
        preliminary += eval_function_info.get_cl_code()
        preliminary += eval_model_func.get_cl_code()

        func_name = 'getObjectiveInstanceValue'
        func = str(preliminary) + '''
            double ''' + func_name + '''(mot_data_struct* data, const mot_float_type* const x, uint observation_index){
                double observation = data->observations[observation_index];
                double model_evaluation = ''' + eval_function_info.get_cl_function_name() + '''(
                    data, x, observation_index);

                ''' + param_listing + '''

                return -''' + eval_model_func.get_cl_function_name() + '''(''' + ','.join(eval_call_args) + ''');
            }
        '''
        return NameFunctionTuple(func_name, func)

    def _get_model_eval_function(self, problems_to_analyze):
        """Get the evaluation function that evaluates the model at the given parameters.

        The returned function should not do any error calculations,
        it should merely return the result of evaluating the model for the given parameters.

        Returns:
            mot.utils.NameFunctionTuple: a named CL function with the following signature:

                .. code-block:: c

                    double <func_name>(mot_data_struct* data, const mot_float_type* const x, uint observation_index);
        """
        composite_model_function = self.get_composite_model_function()

        def get_preliminary():
            cl_preliminary = ''
            cl_preliminary += composite_model_function.get_cl_code()
            pre_model_function = self._get_pre_model_expression_eval_function(problems_to_analyze)
            if pre_model_function:
                cl_preliminary += pre_model_function
            return cl_preliminary

        def get_function_body():
            param_listing = self._get_parameters_listing(
                exclude_list=['{}.{}'.format(m.name, p.name).replace('.', '_') for (m, p) in
                              self._model_functions_info.get_non_model_eval_param_listing()])

            body = ''
            body += dedent(param_listing.replace('\t', ' ' * 4))
            body += self._get_pre_model_expression_eval_code(problems_to_analyze) or ''
            body += '\n'
            body += 'return ' + self._get_composite_model_function_signature(composite_model_function,
                                                                             problems_to_analyze) + ';'
            return body

        function_name = '_evaluateModel'

        cl_function = '''
            double {function_name}(
                    mot_data_struct* data,
                    const mot_float_type* const x,
                    uint observation_index){{

                {body}
            }}
        '''.format(function_name=function_name, body=indent(get_function_body(), ' ' * 4 * 4)[4 * 4:])
        cl_function = dedent(cl_function.replace('\t', ' ' * 4))

        return_str = get_preliminary() + cl_function
        return NameFunctionTuple(function_name, return_str)

    def _get_parameter_transformations(self):
        dec_func_list = []
        enc_func_list = []
        for m, p in self._model_functions_info.get_estimable_parameters_list():
            name = '{}.{}'.format(m.name, p.name)
            parameter = p
            ind = self._model_functions_info.get_parameter_estimable_index(m, p)
            transform = parameter.parameter_transform

            lower_bound = self._get_bound_definition(name, 'lower')
            upper_bound = self._get_bound_definition(name, 'upper')

            s = '{0}[' + str(ind) + '] = ' + transform.get_cl_decode().create_assignment(
                '{0}[' + str(ind) + ']', lower_bound, upper_bound) + ';'

            dec_func_list.append(s)

            s = '{0}[' + str(ind) + '] = ' + transform.get_cl_encode().create_assignment(
                '{0}[' + str(ind) + ']', lower_bound, upper_bound) + ';'

            enc_func_list.append(s)

        return tuple(reversed(enc_func_list)), dec_func_list

    def _get_bound_definition(self, parameter_name, bound_type):
        """Get the definition of the lower bound to use in model functions.

        Since the lower bounds are not added to the ``data`` structure if they are all equal, we have a variable
        way of referencing the lower bound.

        Args:
            parameter_name (str): the name of the parameter as ``<model>.<parameter>``.
            bound_type (str): either ``upper`` or ``lower``.

        Returns:
            str: the way to reference the bound
        """
        if bound_type == 'lower':
            if all_elements_equal(self._lower_bounds[parameter_name]):
                return str(get_single_value(self._lower_bounds[parameter_name]))
            else:
                return 'data->lb_{}[0]'.format(parameter_name.replace('.', '_'))
        else:
            if all_elements_equal(self._upper_bounds[parameter_name]):
                return str(get_single_value(self._upper_bounds[parameter_name]))
            else:
                return 'data->ub_{}[0]'.format(parameter_name.replace('.', '_'))

    def _get_max_numdiff_step(self):
        """Get the numerical differentiation step for each parameter.

        Returns:
            list[float]: for each free parameter the numerical differentiation step size to use
        """
        return [p.numdiff_info.max_step for _, p in self._model_functions_info.get_estimable_parameters_list()]

    def _get_numdiff_scaling_factors(self):
        """Get the parameter scaling factor for each parameter.

        Returns:
            list[float]: for each parameter the scaling factor to use.
        """
        return [p.numdiff_info.scaling_factor for _, p in self._model_functions_info.get_estimable_parameters_list()]

    def _get_numdiff_use_bounds(self):
        """Get the boolean array indicating the use the of the bounds when taking the numerical derivative.

        Returns:
            list[bool]: a list with booleans, with True if we should use the bounds for that parameter, and False
                if we don't have to.
        """
        return [p.numdiff_info.use_bounds for _, p in self._model_functions_info.get_estimable_parameters_list()]

    def _get_numdiff_use_upper_bounds(self):
        """Check for each parameter if we should be using the upper bounds when taking the derivative.

        This is only used if use_bounds is True for a parameter.

        Returns:
            list[bool]: per parameter a boolean to identify if we should use the upper bounds for that parameter.
        """
        return [p.numdiff_info.use_upper_bound for _, p in self._model_functions_info.get_estimable_parameters_list()]

    def _get_numdiff_use_lower_bounds(self):
        """Check for each parameter if we should be using the lower bounds when taking the derivative.

        This is only used if use_bounds is True for a parameter.

        Returns:
            list[bool]: per parameter a boolean to identify if we should use the lower bounds for that parameter.
        """
        return [p.numdiff_info.use_lower_bound for _, p in self._model_functions_info.get_estimable_parameters_list()]

    def _get_numdiff_param_transform(self):
        """Get the parameter transformation for use in the numerical differentiation algorithm.

        Returns:
            mot.utils.NameFunctionTuple: A function with the signature:
                .. code-block:: c

                    void <func_name>(mot_data_struct* data, mot_float_type* params);

                Where the data is the kernel data struct and params is the vector with the suggested parameters and
                which can be modified in place. Note that this is called two times, one with the parameters plus
                the step and one time without.
        """
        transforms = []
        for ind, (_, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
            if p.numdiff_info.modulus is not None and p.numdiff_info.modulus > 0:
                transforms.append(
                    'params[{ind}] = (params[{ind}] - '
                    '({div} * floor(params[{ind}] / {div})));'.format(ind=ind, div=p.numdiff_info.modulus))

        func_name = 'param_transform'
        func = '''
            void {func_name}(mot_data_struct* data, mot_float_type* params){{
                {transforms}
            }}
        '''.format(func_name=func_name, transforms='\n'.join(transforms))
        return NameFunctionTuple(func_name, func)


    def _transform_observations(self, observations):
        """Apply a transformation on the observations before fitting.

        This function is called by get_problems_var_data() just before the observations are handed over to the
        CL routine.

        To implement any behaviour here, you can override this function and add behaviour that changes the observations.

        Args:
            observations (ndarray): the 2d matrix with the observations. This is the list of
                observations used to build the model (that is, *after* the list has been optionally
                limited with problems_to_analyze).

        Returns:
            observations (ndarray): a 2d matrix of the same shape as the input. This should hold the transformed data.
        """
        return observations

    def _get_composite_model_function_signature(self, composite_model, problems_to_analyze):
        """Create the parameter call code for the composite model.

        Args:
            composite_model (CompositeModelFunction): the composite model function
            problems_to_analyze (list): the problems we are analyzing in this round
        """
        param_list = []
        for model, param in composite_model.get_model_parameter_list():
            param_name = '{}.{}'.format(model.name, param.name).replace('.', '_')

            if isinstance(param, ProtocolParameter):
                param_list.append(param.name)
            elif isinstance(param, StaticMapParameter):
                static_map_value = self._get_static_map_value(model, param, problems_to_analyze)

                if all_elements_equal(static_map_value):
                    param_list.append(str(get_single_value(static_map_value)))
                else:
                    pointer_index = '0'
                    if len(static_map_value.shape) > 1 and static_map_value.shape[1] != 1 \
                        and static_map_value.shape[1] == self.get_nmr_inst_per_problem():
                            pointer_index = 'observation_index'
                    param_list.append('data->{}[{}]'.format(param_name, pointer_index))
            elif isinstance(param, CurrentObservationParam):
                param_list.append('data->observations[observation_index]')
            else:
                param_list.append(param_name)

        return composite_model.get_cl_function_name() + '(' + ', '.join(param_list) + ')'

    def _get_parameters_listing(self, exclude_list=()):
        """Get the CL code for the parameter listing, this goes on top of the evaluate function.

        Args:
            exclude_list: an optional list containing parameters to exclude from the listing.
             This should contain full parameter names like: <model_name>_<param_name>

        Returns:
            An CL string that contains all the parameters as primitive data types.
        """
        func = ''
        func += self._get_protocol_parameters_listing(exclude_list=exclude_list)
        func += self._get_fixed_parameters_listing(exclude_list=exclude_list)
        func += self._get_estimable_parameters_listing(exclude_list=exclude_list)
        func += self._get_dependent_parameters_listing(exclude_list=exclude_list)
        return str(func)

    def _get_estimable_parameters_listing(self, exclude_list=()):
        """Get the parameter listing for the free parameters.

        Args:
            exclude_list: a list of parameters to exclude from this listing
        """
        param_list = self._model_functions_info.get_estimable_parameters_list(exclude_priors=True)

        func = ''
        estimable_param_counter = 0
        for m, p in param_list:
            name = '{}.{}'.format(m.name, p.name).replace('.', '_')
            if name not in exclude_list:
                data_type = p.data_type.declaration_type
                assignment = 'x[' + str(estimable_param_counter) + ']'
                func += "\t"*4 + data_type + ' ' + name + ' = ' + assignment + ';' + "\n"
                estimable_param_counter += 1
        return func

    def _get_protocol_parameters_listing(self, exclude_list=()):
        """Get the parameter listing for the protocol parameters.

        Args:
            exclude_list: a list of parameters to exclude from this listing
        """
        param_list = self._model_functions_info.get_protocol_parameters_list()
        const_params_seen = []
        func = ''
        for m, p in param_list:
            if ('{}.{}'.format(m.name, p.name).replace('.', '_')) not in exclude_list:

                param_value = self._input_data.get_input_data(p.name)
                data_type = p.data_type.declaration_type

                if p.name not in const_params_seen:
                    if all_elements_equal(param_value):
                        if p.data_type.is_vector_type:
                            vector_length = p.data_type.vector_length
                            values = [str(val) for val in param_value[0]]
                            if len(values) < vector_length:
                                values.append(str(0))
                            assignment = '(' + data_type + ')(' + ', '.join(values) + ')'
                        else:
                            assignment = str(float(get_single_value(param_value)))
                    else:
                        assignment = 'data->' + p.name + '[observation_index]'
                    func += "\t"*4 + data_type + ' ' + p.name + ' = ' + assignment + ';' + "\n"
                    const_params_seen.append(p.name)
        return func

    def _get_fixed_parameters_listing(self, exclude_list=()):
        """Get the parameter listing for the fixed parameters.

        Args:
            exclude_list: a list of parameters to exclude from this listing
        """
        param_list = self._model_functions_info.get_value_fixed_parameters_list(exclude_priors=True)

        func = ''
        for m, p in param_list:
            name = '{}.{}'.format(m.name, p.name).replace('.', '_')
            if name not in exclude_list:
                data_type = p.data_type.raw_data_type
                value = self._model_functions_info.get_parameter_value('{}.{}'.format(m.name, p.name))
                param_name = '{}.{}'.format(m.name, p.name).replace('.', '_')

                if all_elements_equal(value):
                    assignment = '(' + data_type + ')' + str(float(get_single_value(value)))
                else:
                    assignment = '(' + data_type + ') data->{}[0]'.format(param_name)

                func += "\t"*4 + data_type + ' ' + name + ' = ' + assignment + ';' + "\n"
        return func

    def _get_dependent_parameters_listing(self, dependent_param_list=None, exclude_list=()):
        """Get the parameter listing for the dependent parameters.

        Args:
            dependent_param_list: the list list of dependent params
            exclude_list: a list of parameters to exclude from this listing, note that this will only exclude the
                definition of the parameter, not the dependency code.
        """
        if dependent_param_list is None:
            dependent_param_list = self._model_functions_info.get_dependency_fixed_parameters_list(exclude_priors=True)

        func = ''
        for m, p in dependent_param_list:
            dependency = self._model_functions_info.get_parameter_value('{}.{}'.format(m.name, p.name))

            if dependency.pre_transform_code:
                func += "\t"*4 + self._convert_parameters_dot_to_bar(dependency.pre_transform_code)

            assignment = self._convert_parameters_dot_to_bar(dependency.assignment_code)
            name = '{}.{}'.format(m.name, p.name).replace('.', '_')
            data_type = p.data_type.raw_data_type

            if ('{}.{}'.format(m.name, p.name).replace('.', '_')) not in exclude_list:
                func += "\t"*4 + data_type + ' ' + name + ' = ' + assignment + ';' + "\n"
        return func

    def _get_fixed_parameters_as_var_data(self, problems_to_analyze):
        var_data_dict = {}
        for m, p in self._model_functions_info.get_value_fixed_parameters_list():
            value = self._model_functions_info.get_parameter_value('{}.{}'.format(m.name, p.name))
            param_name = '{}.{}'.format(m.name, p.name).replace('.', '_')

            if not all_elements_equal(value):
                if problems_to_analyze is not None:
                    value = value[problems_to_analyze, ...]
                var_data_dict[param_name] = KernelInputArray(value, ctype=p.data_type.declaration_type)
        return var_data_dict

    def _get_static_parameters_as_var_data(self, problems_to_analyze):
        static_data_dict = {}

        for m, p in self._model_functions_info.get_static_parameters_list():
            value = self._get_static_map_value(m, p, problems_to_analyze)
            param_name = '{}.{}'.format(m.name, p.name).replace('.', '_')

            if not all_elements_equal(value):
                static_data_dict.update({param_name: KernelInputArray(value, ctype=p.data_type.declaration_type)})

        return static_data_dict

    def _get_bounds_as_var_data(self, problems_to_analyze):
        bounds_dict = {}

        for m, p in self._model_functions_info.get_free_parameters_list():
            lower_bound = self._lower_bounds['{}.{}'.format(m.name, p.name)]
            upper_bound = self._upper_bounds['{}.{}'.format(m.name, p.name)]

            for bound_type, value in zip(('lb', 'ub'), (lower_bound, upper_bound)):
                name = bound_type + '_' + '{}.{}'.format(m.name, p.name).replace('.', '_')
                data = value

                if not all_elements_equal(value):
                    if problems_to_analyze is not None:
                        data = data[problems_to_analyze, ...]
                    bounds_dict.update({name: KernelInputArray(data, ctype=p.data_type.declaration_type)})

        return bounds_dict

    def _get_static_map_value(self, model, parameter, problems_to_analyze):
        """Get the map value for the given parameter of the given model.

        The resolution order is as follows, with a latter stage taking preference over an earlier stage

        1. the value defined in the parameter definition
        2. the <param_name> in the input data
        3. the <model_name>.<param_name> in the input data
        4. the <param_name> in the provided initial values
        5. the <model_name>.<param_name> in the provided initial values
        6. the <param_name> in the provided fixed values
        7. the <model_name>.<param_name> in the provided fixed values

        This only returns the problems for which problems_to_analyze is set.

        Args:
            model (ModelCLFunction): the model function
            parameter (CLParameter): the parameter for which we want to get the value
            problems_to_analyze (ndarray): the problems we are interested in

        Returns:
            ndarray or number: the value for the given parameter.
        """
        def resolve_value():
            value = self._model_functions_info.get_parameter_value('{}.{}'.format(model.name, parameter.name))

            if self._input_data.get_input_data(parameter.name) is not None:
                value = self._input_data.get_input_data(parameter.name)

            if self._input_data.get_input_data('{}.{}'.format(model.name, parameter.name)) is not None:
                value = self._input_data.get_input_data('{}.{}'.format(model.name, parameter.name))

            return np.squeeze(value)

        param_data = resolve_value()

        if param_data is None:
            raise ValueError('No suitable data could be found for the static parameter {}.'.format(parameter.name))

        if is_scalar(param_data):
            return get_single_value(param_data)

        if param_data.shape[0] == self._input_data.nmr_problems:
            if problems_to_analyze is not None:
                return param_data[problems_to_analyze, ...]

        return param_data

    def _get_param_listing_for_param(self, m, p):
        """Get the param listing for one specific parameter. This can be used for example for the noise model params.
        """
        data_type = p.data_type.raw_data_type
        name = '{}.{}'.format(m.name, p.name).replace('.', '_')
        assignment = ''

        if isinstance(p, ProtocolParameter):
            assignment = 'data->' + p.name + '[observation_index]'
        elif isinstance(p, FreeParameter):
            assignment = self._get_free_parameter_assignment_value(m, p)

        return data_type + ' ' + name + ' = ' + assignment + ';' + "\n"

    def _get_free_parameter_assignment_value(self, m, p):
        """Get the assignment value for one of the free parameters.

        Since the free parameters can be fixed we need an auxiliary routine to get the assignment value.

        Args:
            m: model
            p: parameter
        """
        data_type = p.data_type.raw_data_type
        value = self._model_functions_info.get_parameter_value('{}.{}'.format(m.name, p.name))

        assignment = ''

        if self._model_functions_info.is_fixed_to_value('{}.{}'.format(m.name, p.name)):
            param_name = '{}.{}'.format(m.name, p.name).replace('.', '_')
            if all_elements_equal(value):
                assignment = '(' + data_type + ')' + str(float(get_single_value(value)))
            else:
                assignment = '(' + data_type + ') data->{}[0]'.format(param_name)
        elif self._model_functions_info.is_fixed_to_dependency(m, p):
            return self._get_dependent_parameters_listing(((m, p),))
        else:
            ind = self._model_functions_info.get_parameter_estimable_index(m, p)
            assignment += 'x[' + str(ind) + ']'

        return assignment

    def _convert_parameters_dot_to_bar(self, string):
        """Convert a string containing parameters with . to parameter names with _"""
        for m, p in self._model_functions_info.get_model_parameter_list():
            dname = '{}.{}'.format(m.name, p.name)
            bname = '{}.{}'.format(m.name, p.name).replace('.', '_')
            string = string.replace(dname, bname)
        return string

    def _init_fixed_duplicates_dependencies(self):
        """Find duplicate fixed parameters, and make dependencies of them. This saves data transfer in CL."""
        var_data_dict = {}
        for m, p in self._model_functions_info.get_free_parameters_list():
            param_name = '{}.{}'.format(m.name, p.name)
            if self._model_functions_info.is_fixed_to_value(param_name):
                value = self._model_functions_info.get_parameter_value(param_name)

                if not is_scalar(value):
                    duplicate_found = False
                    duplicate_key = None

                    for key, data in var_data_dict.items():
                        if np.array_equal(data, value):
                            duplicate_found = True
                            duplicate_key = key
                            break

                    if duplicate_found:
                        self.fix(param_name, SimpleAssignment(duplicate_key))
                    else:
                        var_data_dict.update({param_name: value})

    def _get_observations_data(self, problems_to_analyze):
        """Get the observations to use in the kernel.

        Can return None if there are no observations.
        """
        observations = self._input_data.observations
        if observations is not None:
            if problems_to_analyze is not None:
                observations = observations[problems_to_analyze, ...]
            observations = self._transform_observations(observations)
            return {'observations': KernelInputArray(observations)}
        return {}

    def _get_protocol_data(self, problems_to_analyze):
        return_data = {}

        for m, p in self._model_functions_info.get_model_parameter_list():
            if isinstance(p, ProtocolParameter):
                value = self._input_data.get_input_data(p.name)
                if not all_elements_equal(value):
                    if value.shape[0] == self._input_data.nmr_problems:
                        if problems_to_analyze is not None:
                            value = value[problems_to_analyze, ...]
                        const_d = {p.name: KernelInputArray(value, ctype=p.data_type.declaration_type)}
                    else:
                        const_d = {p.name: KernelInputArray(value, ctype=p.data_type.declaration_type, offset_str='0')}
                    return_data.update(const_d)
        return return_data

    def _get_pre_model_expression_eval_code(self, problems_to_analyze):
        """The code called in the evaluation function.

        This is called after the parameters are initialized and before the model signal expression. It can call
        functions defined in _get_pre_model_expression_eval_function()

        Returns:
            str: cl code containing evaluation changes,
        """
        return ''

    def _get_pre_model_expression_eval_function(self, problems_to_analyze):
        """Function used in the model evaluation generation function.

        The idea is that some implementing models may need to change some of the protocol or fixed parameters
        before they are handed over to the signal expression function. This function is called by the
        get_model_eval_function function during model evaluation function construction.

        Returns:
            str: cl function to be used in conjunction with the output of the function
                _get_pre_model_expression_eval_model()
        """

    def _set_default_dependencies(self):
        """Initialize the default dependencies.

        By default this adds dependencies for the fixed data that is used in multiple parameters.
        Additionally, if enforce weights sum to one is set, this adds the dependency on the first weight.
        """
        self._init_fixed_duplicates_dependencies()
        if self._enforce_weights_sum_to_one:
            names = ['{}.{}'.format(m.name, p.name) for (m, p) in self._model_functions_info.get_weights()]
            if len(names) > 1:
                self.fix(names[0], SimpleAssignment('max((double)1 - ({}), (double)0)'.format(' + '.join(names[1:]))))

    def _get_weight_sum_to_one_transformation(self):
        """Returns a snippet of CL for the encode and decode functions to force the sum of the weights to 1"""
        weight_indices = []
        for (m, p) in self._model_functions_info.get_estimable_weights():
            weight_indices.append(self._model_functions_info.get_parameter_estimable_index(m, p))

        if len(weight_indices) > 1:
            return '''
                mot_float_type _weight_sum = ''' + ' + '.join('x[{}]'.format(index) for index in weight_indices) + ''';
                if(_weight_sum > 1.0){
                    ''' + '\n'.join('x[{}] /= _weight_sum;'.format(index) for index in weight_indices) + '''
                }
            '''
        return ''


class SampleModelBuilder(OptimizeModelBuilder):

    def __init__(self, model_name, model_tree, likelihood_function, signal_noise_model=None, input_data=None,
                 enforce_weights_sum_to_one=True):
        """Create a new model builder for sampling purposes.

        Attributes:
            model_priors (list[mot.cl_function.CLFunction]): the list of model priors this class
                will also use (next to the priors defined in the parameters).
        """
        super(SampleModelBuilder, self).__init__(model_name, model_tree, likelihood_function, signal_noise_model,
                                                 input_data=input_data,
                                                 enforce_weights_sum_to_one=enforce_weights_sum_to_one)

        self._model_priors = []

        if self._enforce_weights_sum_to_one:
            weight_prior = self._get_weight_prior()
            if weight_prior:
                self._model_priors.append(weight_prior)

        for compartment in self._model_functions_info.get_model_list():
            priors = compartment.get_model_function_priors()
            if priors:
                for prior in priors:
                    self._model_priors.append(_ModelFunctionPriorToCompositeModelPrior(prior, compartment.name))

    def _init_model_information_container(self, model_tree, likelihood_function, signal_noise_model):
        """Get the model information container object.

        This is called in the __init__ to provide the new model with the correct subclass function information
        object. The rationale is that some subclasses may have additional parameters not present in optimization. For
        example, in sampling one can have priors with parameters. These parameters must be added to the model and the
        best point to do that is in the ModelFunctionsInformation object.

        Returns:
            ModelFunctionsInformation: the model function information object
        """
        return ModelFunctionsInformation(model_tree, likelihood_function, signal_noise_model,
                                         enable_prior_parameters=True)

    def build(self, problems_to_analyze=None):
        """Construct the final immutable model with the current settings.

        Returns:
            OptimizeModelInterface: an implementation an optimization model with all the current settings

        Raises:
            RuntimeError: if some of the required items are not set prior to building.
        """
        simple_optimize_model = super(SampleModelBuilder, self).build(problems_to_analyze)
        return SimpleSampleModel(simple_optimize_model,
                                 self._get_log_likelihood_per_observation_function(problems_to_analyze),
                                 self._get_log_prior_function_builder(),
                                 self._get_finalize_proposal_function_builder())

    def _get_log_prior_function_builder(self):
        def get_preliminary():
            cl_str = ''
            for i, (m, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
                cl_str += p.sampling_prior.get_cl_code()

            for model_prior in self._model_priors:
                cl_str += model_prior.get_cl_code()
            return cl_str

        def get_body():
            cl_str = ''
            for i, (m, p) in enumerate(self._model_functions_info.get_estimable_parameters_list()):
                name = '{}.{}'.format(m.name, p.name)

                if all_elements_equal(self._lower_bounds[name]):
                    if np.isneginf(get_single_value(self._lower_bounds[name])):
                        lower_bound = '-INFINITY'
                    else:
                        lower_bound = str(get_single_value(self._lower_bounds[name]))
                else:
                    lower_bound = 'data->lb_' + name.replace('.', '_') + '[0]'

                if all_elements_equal(self._upper_bounds[name]):
                    if np.isposinf(get_single_value(self._upper_bounds[name])):
                        upper_bound = 'INFINITY'
                    else:
                        upper_bound = str(get_single_value(self._upper_bounds[name]))
                else:
                    upper_bound = 'data->ub_' + name.replace('.', '_') + '[0]'

                function_name = p.sampling_prior.get_cl_function_name()

                if m.get_prior_parameters(p):
                    prior_params = []
                    for prior_param in m.get_prior_parameters(p):
                        if self._model_functions_info.is_parameter_estimable(m, prior_param):
                            estimable_index = self._model_functions_info.get_parameter_estimable_index(m, prior_param)
                            prior_params.append('x[{}]'.format(estimable_index))
                        else:
                            value = self._model_functions_info.get_parameter_value(
                                '{}.{}'.format(m.name, prior_param.name))
                            if all_elements_equal(value):
                                prior_params.append(str(get_single_value(value)))
                            else:
                                prior_params.append('data->' +
                                                    '{}.{}'.format(m.name, prior_param.name).replace('.', '_')) + '[0]'

                    cl_str += 'prior *= {}(x[{}], {}, {}, {});\n'.format(function_name, i, lower_bound, upper_bound,
                                                                         ', '.join(prior_params))
                else:
                    cl_str += 'prior *= {}(x[{}], {}, {});\n'.format(function_name, i, lower_bound, upper_bound)

            for model_prior in self._model_priors:
                function_name = model_prior.get_cl_function_name()
                parameters = []

                for param in model_prior.get_parameters():
                    assignment_value = self._get_free_parameter_assignment_value(
                        *self._model_functions_info.get_model_parameter_by_name(param.name))
                    parameters.append(assignment_value)

                cl_str += '\tprior *= {}({});\n'.format(function_name, ', '.join(parameters))

            cl_str += '\n\treturn log(prior);'
            return cl_str

        preliminary = get_preliminary()
        body = get_body()

        def builder(address_space_parameter_vector):
            func_name = 'getLogPrior'
            prior = '''
                {preliminary}

                mot_float_type {func_name}(mot_data_struct* data,
                                           {address_space_parameter_vector} const mot_float_type* const x){{

                    mot_float_type prior = 1.0;

                    {body}
                }}
                '''.format(func_name=func_name, address_space_parameter_vector=address_space_parameter_vector,
                           preliminary=preliminary, body=body)
            return NameFunctionTuple(func_name, prior)
        return builder

    def _get_log_likelihood_per_observation_function(self, problems_to_analyze):
        eval_function_info = self._get_model_eval_function(problems_to_analyze)
        eval_function_signature = self._likelihood_function.get_log_likelihood_function()

        eval_call_args = ['observation', 'model_evaluation']
        param_listing = ''
        for p in eval_function_signature.get_free_parameters():
            param_listing += self._get_param_listing_for_param(eval_function_signature, p)
            eval_call_args.append('{}.{}'.format(eval_function_signature.name, p.name).replace('.', '_'))

        preliminary = ''
        preliminary += eval_function_info.get_cl_code()

        eval_model_func = self._likelihood_function.get_log_likelihood_function()

        func_name = 'getLogLikelihoodPerObservation'
        func = str(preliminary) + eval_model_func.get_cl_code() + '''
           double ''' + func_name + '''(mot_data_struct* data, const mot_float_type* const x,
                                        uint observation_index){

               double observation = data->observations[observation_index];
               double model_evaluation = ''' + eval_function_info.get_cl_function_name() + '''(
                   data, x, observation_index);

               ''' + param_listing + '''

               return ''' + eval_model_func.get_cl_function_name() + '''(''' + ','.join(eval_call_args) + ''');
           }
       '''
        return NameFunctionTuple(func_name, func)

    def _get_weight_prior(self):
        """Get the prior limiting the weights between 0 and 1"""
        weights = []
        for (m, p) in self._model_functions_info.get_estimable_weights():
            weights.append(('mot_float_type', '{}.{}'.format(m.name, p.name)))

        if len(weights) > 1:
            return SimpleCLFunction(
                'mot_float_type', 'prior_estimable_weights_sum_to_one',
                weights, 'return (' + ' + '.join(el[1].replace('.', '_') for el in weights) + ') <= 1;')
        return None

    def _get_finalize_proposal_function_builder(self):
        """Get the building function used to finalize the proposal"""
        def builder(address_space_parameter_vector):
            func_name = 'finalizeProposal'
            prior = '''
                {preliminary}
                
                mot_float_type {func_name}(mot_data_struct* data,
                                           {address_space_parameter_vector} mot_float_type* x){{
                    {body}
                }}
                '''.format(func_name=func_name, address_space_parameter_vector=address_space_parameter_vector,
                           preliminary='', body='')
            return NameFunctionTuple(func_name, prior)
        return builder


class CompositeModelFunction(SimpleCLFunction):

    def __init__(self, model_tree, signal_noise_model=None):
        """The model function for the total constructed model.

        This combines all the functions in the model tree into one big function and exposes that function and
        its parameters.

        Args:
            model_tree (mdt.model_building.trees.CompartmentModelTree): the model tree object
            signal_noise_model (mdt.model_building.signal_noise_models.SignalNoiseModel): the optional signal
                noise model to use to add noise to the model prediction
        """
        self._model_tree = model_tree
        self._signal_noise_model = signal_noise_model
        self._models = list(self._model_tree.get_compartment_models())
        if self._signal_noise_model:
            self._models.append(self._signal_noise_model)
        self._parameter_model_list = list((m, p) for m in self._models for p in m.get_parameters())

        cl_function_name = '_composite_model_function'

        super(CompositeModelFunction, self).__init__(
            'double', cl_function_name,
            [p.get_renamed(external_name) for m, p, _, external_name in self._get_model_function_parameters()],
            self._get_model_function_body(),
            dependencies=self._models)

    def get_model_parameter_list(self):
        """Get the model and parameter tuples that constructed this composite model.

        This is used by the model builder, to construct the model function call.

        Returns:
            list of tuple: the list of (model, parameter) tuples for each of the models and parameters.
        """
        return [(m, p) for m, p, cl_name, ext_name in self._get_model_function_parameters()]

    def _get_model_function_parameters(self):
        """Get the parameters to use in the model function.

        Returns:
            list of tuples: per parameter a tuple with (model, parameter, cl_name, external_name)
                where the cl_name is how we reference the parameter in the CL code and the external_name is
                how we reference the parameter in a call to for example 'evaluate'.
        """
        seen_shared_params = []

        shared_params = []
        other_params = []

        for m, p in self._parameter_model_list:
            if isinstance(p, (ProtocolParameter, CurrentObservationParam)):
                if p.name not in seen_shared_params:
                    shared_params.append((m, p, p.name, p.name))
                    seen_shared_params.append(p.name)
            else:
                other_params.append((m, p, '{}_{}'.format(m.name, p.name), '{}.{}'.format(m.name, p.name)))
        return shared_params + other_params

    def _get_model_function_body(self):
        """Get the CL code for the body of the model function as build by this model.

        Returns:
            str: the CL code for the body of this code
        """
        def build_model_expression():
            tree = self._build_model_from_tree(self._model_tree, 0)

            model_expression = ''
            if self._signal_noise_model:
                noise_params = ''
                for p in self._signal_noise_model.get_free_parameters():
                    noise_params += '{}.{}'.format(self._signal_noise_model.name, p.name).replace('.', '_')
                model_expression += '{}(({}), {});'.format(self._signal_noise_model.get_cl_function_name(),
                                                           tree, noise_params)
            else:
                model_expression += '(' + tree + ');'
            return model_expression

        return_str = 'return ' + build_model_expression()
        return dedent(return_str.replace('\t', '    '))

    def _build_model_from_tree(self, node, depth):
        """Construct the model equation from the provided model tree.

        Args:
            node: the next to to process
            depth (int): the current tree depth

        Returns:
            str: model (sub-)equation
        """
        def model_to_string(model):
            """Convert a model to CL string."""
            param_list = []
            for param in model.get_parameters():
                if isinstance(param, (ProtocolParameter, CurrentObservationParam)):
                    param_list.append(param.name)
                else:
                    param_list.append('{}.{}'.format(model.name, param.name).replace('.', '_'))
            return model.get_cl_function_name() + '(' + ', '.join(param_list) + ')'

        if not node.children:
            return model_to_string(node.data)
        else:
            subfuncs = []
            for child in node.children:
                if child.children:
                    subfuncs.append(self._build_model_from_tree(child, depth + 1))
                else:
                    subfuncs.append(model_to_string(child.data))

            operator = node.data
            func = (' ' + operator + ' ').join(subfuncs)

        if func[0] == '(':
            return '(' + func + ')'
        return '(' + "\n" + ("\t" * int((depth/2)+5)) + func + "\n" + ("\t" * int((depth/2)+4)) + ')'


class ModelFunctionsInformation(object):

    def __init__(self, model_tree, likelihood_function, signal_noise_model=None, enable_prior_parameters=False):
        """Contains centralized information about the model functions in the model builder parent.

        Args:
            model_tree (mdt.model_building.trees.CompartmentModelTree): the model tree object
            likelihood_function (mdt.model_building.likelihood_functions.LikelihoodFunction): the likelihood function to
                use for the resulting complete model
            signal_noise_model (mdt.model_building.signal_noise_models.SignalNoiseModel): the signal
                noise model to use to add noise to the model prediction
            enable_prior_parameters (boolean): adds possible prior parameters to the list of parameters in the model
        """
        self._model_tree = model_tree
        self._likelihood_function = likelihood_function
        self._signal_noise_model = signal_noise_model
        self._enable_prior_parameters = enable_prior_parameters

        self._model_list = self._get_model_list()
        self._model_parameter_list = self._get_model_parameter_list()
        self._prior_parameters_info = self._get_prior_parameters_info()

        self._check_for_double_model_names()

        self._fixed_parameters = {'{}.{}'.format(m.name, p.name): p.fixed for m, p in
                                  self.get_model_parameter_list() if isinstance(p, FreeParameter)}
        self._fixed_values = {'{}.{}'.format(m.name, p.name): p.value for m, p in self.get_free_parameters_list()}

        self._parameter_values = {'{}.{}'.format(m.name, p.name): p.value for m, p in self.get_model_parameter_list()
                                  if hasattr(p, 'value')}

    def set_parameter_value(self, parameter_name, value):
        """Set the value we will use for the given parameter.

        If the parameter is a fixed free parameter we will set the fixed value to the given value.

        Args:
            parameter_name (string): A model.param name like 'Ball.d'
            value (scalar or vector or string or AbstractParameterDependency): The value or dependency
                to fix the given parameter to. Dependency objects and strings are only value for fixed free parameters.
        """
        if parameter_name in self._fixed_parameters and self._fixed_parameters[parameter_name]:
            self._fixed_values[parameter_name] = value
        else:
            self._parameter_values[parameter_name] = value

    def get_parameter_value(self, parameter_name):
        """Get the parameter value for the given parameter. This is regardless of model fixation.

        Returns:
            float or ndarray: the value for the given parameter
        """
        if parameter_name in self._fixed_parameters and self._fixed_parameters[parameter_name]:
            return self._fixed_values[parameter_name]
        return self._parameter_values[parameter_name]

    def fix_parameter(self, parameter_name, value):
        """Fix the indicated free parameter to the given value.

        Args:
            parameter_name (string): A model.param name like 'Ball.d'
            value (scalar or vector or string or AbstractParameterDependency): The value or dependency
                to fix the given parameter to.
        """
        self._fixed_parameters[parameter_name] = True
        self._fixed_values[parameter_name] = value

    def unfix(self, parameter_name):
        """Unfix the indicated parameter

        Args:
            parameter_name (str): the name of the parameter to fix or unfix
        """
        self._fixed_parameters[parameter_name] = False

    def get_model_list(self):
        """Get the list of all the applicable model functions

        Returns:
            list of mdt.model_building.model_functions.ModelCLFunction: the list of model functions.
        """
        return self._model_list

    def get_model_parameter_list(self):
        """Get a list of all model, parameter tuples.

        Returns:
            list of tuple: the list of tuples containing (model, parameters)
        """
        param_list = copy.copy(self._model_parameter_list)

        if self._enable_prior_parameters:
            for prior_info in self._prior_parameters_info.values():
                if prior_info:
                    param_list.extend(prior_info)

        return param_list

    def get_free_parameters_list(self, exclude_priors=False):
        """Gets the free parameters as (model, parameter) tuples from the model listing.
        This does not incorporate checking for fixed parameters.

        Args:
            exclude_priors (boolean): if we want to exclude the parameters for the priors

        Returns:
            list of tuple: the list of tuples containing (model, parameters)
        """
        free_params = list((m, p) for m, p in self._model_parameter_list if isinstance(p, FreeParameter))

        if not exclude_priors:
            if self._enable_prior_parameters:
                prior_params = []
                for m, p in free_params:
                    prior_params.extend((m, prior_p) for prior_p in m.get_prior_parameters(p)
                                        if self.is_parameter_estimable(m, p) and isinstance(prior_p, FreeParameter))
                free_params.extend(prior_params)

        return free_params

    def get_estimable_parameters_list(self, exclude_priors=False):
        """Gets a list (as model, parameter tuples) of all parameters that are estimable.

        Args:
            exclude_priors (boolean): if we want to exclude the parameters for the priors

        Returns:
            list of tuple: the list of estimable parameters
        """
        estimable_parameters = [(m, p) for m, p in self._model_parameter_list if self.is_parameter_estimable(m, p)]

        if not exclude_priors:
            if self._enable_prior_parameters:
                prior_params = []
                for m, p in estimable_parameters:
                    prior_params.extend((m, prior_p) for prior_p in m.get_prior_parameters(p) if not prior_p.fixed)
                estimable_parameters.extend(prior_params)

        return estimable_parameters

    def get_value_fixed_parameters_list(self, exclude_priors=False):
        """Gets a list (as model, parameter tuples) of all parameters that are fixed to a value.

        Args:
            exclude_priors (boolean): if we want to exclude the parameters for the priors

        Returns:
            list of tuple: the list of value fixed parameters
        """
        value_fixed_parameters = []
        for m, p in self.get_free_parameters_list(exclude_priors=exclude_priors):
            if self.is_fixed_to_value('{}.{}'.format(m.name, p.name)):
                value_fixed_parameters.append((m, p))
        return value_fixed_parameters

    def get_dependency_fixed_parameters_list(self, exclude_priors=False):
        """Gets a list (as model, parameter tuples) of all parameters that are fixed to a dependency.

        Args:
            exclude_priors (boolean): if we want to exclude the parameters for the priors

        Returns:
            list of tuple: the list of value fixed parameters
        """
        dependency_fixed_parameters = []
        for m, p in self.get_free_parameters_list(exclude_priors=exclude_priors):
            if self.is_fixed_to_dependency(m, p):
                dependency_fixed_parameters.append((m, p))
        return dependency_fixed_parameters

    def get_static_parameters_list(self):
        """Gets the static parameters (as model, parameter tuples) from the model listing."""
        static_params = list((m, p) for m, p in self.get_model_parameter_list() if isinstance(p, StaticMapParameter))

        if self._enable_prior_parameters:
            prior_params = []
            for m, p in self.get_estimable_parameters_list():
                prior_params.extend((m, prior_p) for prior_p in m.get_prior_parameters(p)
                                    if isinstance(prior_p, FreeParameter))
            static_params.extend(prior_params)

        return static_params

    def get_protocol_parameters_list(self):
        """Gets the static parameters (as model, parameter tuples) from the model listing."""
        return list((m, p) for m, p in self.get_model_parameter_list() if isinstance(p, ProtocolParameter))

    def get_model_parameter_by_name(self, parameter_name):
        """Get the parameter object of the given full parameter name in dot format.

        Args:
            parameter_name (string): the parameter name in dot format: <model>.<param>

        Returns:
            tuple: containing the (model, parameter) pair for the given parameter name
        """
        for m, p in self.get_model_parameter_list():
            if '{}.{}'.format(m.name, p.name) == parameter_name:
                return m, p
        raise ValueError('The parameter with the name "{}" could not be found in this model.'.format(parameter_name))

    def get_non_model_eval_param_listing(self):
        """Get the model, parameter tuples for all parameters that are not used in the likelihood function.

        Basically this returns the parameters of the likelihood function.

        Returns:
            tuple: the (model, parameter) tuple for all non model evaluation parameters
        """
        listing = []
        for p in self._likelihood_function.get_log_likelihood_function().get_parameters():
            listing.append((self._likelihood_function.get_log_likelihood_function(), p))
        return listing

    def is_fixed(self, parameter_name):
        """Check if the given (free) parameter is fixed or not (either to a value or to a dependency).

        Args:
            parameter_name (str): the name of the parameter to fix or unfix

        Returns:
            boolean: if the parameter is fixed or not (can be fixed to a value or dependency).
        """
        return parameter_name in self._fixed_parameters and self._fixed_parameters[parameter_name]

    def is_fixed_to_value(self, parameter_name):
        """Check if the given (free) parameter is fixed to a value.

        Args:
            parameter_name (str): the name of the parameter to fix or unfix

        Returns:
            boolean: if the parameter is fixed to a value or not
        """
        if self.is_fixed(parameter_name):
            return not isinstance(self._fixed_values[parameter_name], AbstractParameterDependency)
        return False

    def is_fixed_to_dependency(self, model, param):
        """Check if the given model and parameter name combo has a dependency.

        Args:
            model (mdt.model_building.model_functions.ModelCLFunction): the model function
            param (mot.cl_parameter.CLFunctionParameter): the parameter

        Returns:
            boolean: if the given parameter has a dependency
        """
        model_param_name = '{}.{}'.format(model.name, param.name)
        if self.is_fixed(model_param_name):
            return isinstance(self._fixed_values[model_param_name], AbstractParameterDependency)
        return False

    def is_parameter_estimable(self, model, param):
        """Check if the given model parameter is estimable.

        A parameter is estimable if it is of the Free parameter type and is not fixed.

        Args:
            model (mdt.model_building.model_functions.ModelCLFunction): the model function
            param (mot.cl_parameter.CLFunctionParameter): the parameter

        Returns:
            boolean: true if the parameter is estimable, false otherwise
        """
        return isinstance(param, FreeParameter) and not self.is_fixed('{}.{}'.format(model.name, param.name))

    def get_weights(self):
        """Get all the model functions/parameter tuples of the models that are a subclass of WeightType

        Returns:
            list: the list of compartment models that are a subclass of WeightType as (model, parameter) tuples.
        """
        weight_models = [m for m in self._model_tree.get_compartment_models() if isinstance(m, WeightType)]
        weights = []
        for m in weight_models:
            for p in m.get_free_parameters():
                weights.append((m, p))
        return weights

    def get_estimable_weights(self):
        """Get all the estimable weights.

        Returns:
            list of tuples: the list of compartment models/parameter pairs for models that are a subclass of WeightType
        """
        return [(m, p) for m, p in self.get_weights() if self.is_parameter_estimable(m, p)]

    def _get_model_parameter_list(self):
        """Get a list of all model, parameter tuples.

        Returns:
            list of tuple: the list of tuples containing (model, parameters)
        """
        return list((m, p) for m in self._model_list for p in m.get_parameters())

    def _get_prior_parameters_info(self):
        """Get a dictionary with the prior parameters for each of the model parameters.

        Returns:
            dict: lookup dictionary matching model names to parameter lists
        """
        prior_lookup_dict = {}
        for model in self._model_list:
            for param in model.get_free_parameters():
                prior_lookup_dict.update({
                    '{}.{}'.format(model.name, param.name): list((model, p) for p in model.get_prior_parameters(param))
                })
        return prior_lookup_dict

    def get_parameter_estimable_index(self, model, param):
        """Get the index of this parameter in the parameters list

        This returns the position of this parameter in the 'x', parameter vector in the CL kernels.

        Args:
            model (mdt.model_building.model_functions.ModelCLFunction): the model function
            param (mot.cl_parameter.CLFunctionParameter): the parameter

        Returns:
            int: the index of the requested parameter in the list of optimized parameters

        Raises:
            ValueError: if the given parameter could not be found as an estimable parameter.
        """
        ind = 0
        for m, p in self.get_estimable_parameters_list():
            if m.name == model.name and p.name == param.name:
                return ind
            ind += 1
        raise ValueError('The given estimable parameter "{}" could not be found in this model'.format(
            '{}.{}'.format(model.name, param.name)))

    def get_parameter_estimable_index_by_name(self, model_param_name):
        """Get the index of this parameter in the parameters list

        This returns the position of this parameter in the 'x', parameter vector in the CL kernels.

        Args:
            model_param_name (str): the model parameter name

        Returns:
            int: the index of the requested parameter in the list of optimized parameters

        Raises:
            ValueError: if the given parameter could not be found as an estimable parameter.
        """
        ind = 0
        for m, p in self.get_estimable_parameters_list():
            if '{}.{}'.format(m.name, p.name) == model_param_name:
                return ind
            ind += 1
        raise ValueError('The given estimable parameter "{}" could not be found in this model'.format(model_param_name))

    def has_parameter(self, model_param_name):
        """Check to see if the given parameter is defined in this model.

        Args:
            model_param_name (string): A model.param name like 'Ball.d'

        Returns:
            boolean: true if the parameter is defined in this model, false otherwise.
        """
        for m, p in self.get_model_parameter_list():
            if '{}.{}'.format(m.name, p.name) == model_param_name:
                return True
        return False

    def _get_model_list(self):
        """Get the list of all the applicable model functions"""
        models = list(self._model_tree.get_compartment_models())
        models.append(self._likelihood_function.get_log_likelihood_function())
        if self._signal_noise_model:
            models.append(self._signal_noise_model)
        return models

    def _check_for_double_model_names(self):
        models = self._model_list
        model_names = []
        for m in models:
            if m.name in model_names:
                raise DoubleModelNameException("Double model name detected in the model tree.", m.name)
            model_names.append(m.name)


class ParameterTransformedModel(OptimizeModelInterface):

    def __init__(self, model, parameter_codec):
        """Decorates the given model with parameter encoding and decoding transformations.

        This decorates a few of the given function calls with the right parameter encoding and decoding transformations
        such that both the underlying model and the calling routines are unaware that the parameters have been altered.

        Args:
            model (OptimizeModelInterface): the model to decorate
            parameter_codec (mdt.model_building.utils.ParameterCodec): the parameter codec to use
        """
        self._model = model
        self._parameter_codec = parameter_codec

    def decode_parameters(self, parameters):
        """Decode the given parameters back to model space.

        Args:
            parameters (ndarray): the parameters to transform back to model space
        """
        space_transformer = CodecRunner()
        return space_transformer.decode(parameters, self.get_kernel_data(), self._parameter_codec)

    def encode_parameters(self, parameters):
        """Decode the given parameters into optimization space

        Args:
            parameters (ndarray): the parameters to transform into optimization space
        """
        space_transformer = CodecRunner()
        return space_transformer.encode(parameters, self.get_kernel_data(), self._parameter_codec)

    @property
    def name(self):
        return self._model.name

    def get_kernel_data(self):
        return self._model.get_kernel_data()

    def get_nmr_problems(self):
        return self._model.get_nmr_problems()

    def get_nmr_inst_per_problem(self):
        return self._model.get_nmr_inst_per_problem()

    def get_nmr_parameters(self):
        return self._model.get_nmr_parameters()

    def get_pre_eval_parameter_modifier(self):
        old_modifier = self._model.get_pre_eval_parameter_modifier()
        new_fname = 'wrapped_' + old_modifier.get_cl_function_name()

        code = old_modifier.get_cl_code()
        code += self._parameter_codec.get_parameter_decode_function('_decodeParameters')
        code += '''
            void ''' + new_fname + '''(mot_data_struct* data, mot_float_type* x){
                _decodeParameters(data, x);
                ''' + old_modifier.get_cl_function_name() + '''(data, x);
            }
        '''
        return NameFunctionTuple(new_fname, code)

    def get_objective_per_observation_function(self):
        return self._model.get_objective_per_observation_function()

    def get_lower_bounds(self):
        # todo add codec transform here
        return self._model.get_lower_bounds()

    def get_upper_bounds(self):
        # todo add codec transform here
        return self._model.get_upper_bounds()

    def finalize_optimized_parameters(self, parameters):
        return self._model.finalize_optimized_parameters(self.decode_parameters(parameters))

    def __getattr__(self, item):
        return getattr(self._model, item)


class ParameterNameException(Exception):
    """Thrown when the a parameter of an given name could not be found."""
    pass


class DoubleModelNameException(Exception):
    """Thrown when there are two models with the same name."""
    pass


class _ModelFunctionPriorToCompositeModelPrior(SimpleCLFunction):

    def __init__(self, model_function_prior, compartment_name):
        """Simple prior class for easily converting the compartment priors to composite model priors."""
        parameters = [SimpleCLFunctionParameter('mot_float_type', '{}.{}'.format(compartment_name, p.name))
                      for p in model_function_prior.get_parameters()]
        self._old_params = model_function_prior.get_parameters()

        super(_ModelFunctionPriorToCompositeModelPrior, self).__init__(
            model_function_prior.get_return_type(),
            model_function_prior.get_cl_function_name(),
            parameters,
            model_function_prior.get_cl_body(),
            dependencies=model_function_prior.get_dependencies(),
            cl_extra=model_function_prior.get_cl_extra()
        )

    def _get_parameter_signatures(self):
        return ['{} {}'.format(p.data_type.get_declaration(), p.name.replace('.', '_')) for p in self._old_params]


class SimpleOptimizeModel(NumericalDerivativeInterface):

    def __init__(self, used_problem_indices,
                 name, kernel_data_info, nmr_problems, nmr_inst_per_problem,
                 nmr_estimable_parameters, initial_parameters, pre_eval_parameter_modifier,
                 objective_per_observation_function, lower_bounds, upper_bounds, numdiff_step,
                 numdiff_scaling_factors, numdiff_use_bounds, numdiff_use_lower_bounds,
                 numdiff_use_upper_bounds, numdiff_param_transform):
        self.used_problem_indices = used_problem_indices
        self.name = name
        self._kernel_data_info = kernel_data_info
        self._nmr_problems = nmr_problems
        self._nmr_inst_per_problem = nmr_inst_per_problem
        self._nmr_estimable_parameters = nmr_estimable_parameters
        self._initial_parameters = initial_parameters
        self._pre_eval_parameter_modifier = pre_eval_parameter_modifier
        self._objective_per_observation_function = objective_per_observation_function
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        self._numdiff_step = numdiff_step
        self._numdiff_scaling_factors = numdiff_scaling_factors
        self._numdiff_use_bounds = numdiff_use_bounds
        self._numdiff_use_lower_bounds = numdiff_use_lower_bounds
        self._numdiff_use_upper_bounds = numdiff_use_upper_bounds
        self._numdiff_param_transform = numdiff_param_transform

    def get_kernel_data(self):
        return self._kernel_data_info

    def get_nmr_problems(self):
        return self._nmr_problems

    def get_nmr_inst_per_problem(self):
        return self._nmr_inst_per_problem

    def get_nmr_parameters(self):
        return self._nmr_estimable_parameters

    def get_pre_eval_parameter_modifier(self):
        return self._pre_eval_parameter_modifier

    def get_objective_per_observation_function(self):
        return self._objective_per_observation_function

    def get_initial_parameters(self):
        return self._initial_parameters

    def get_lower_bounds(self):
        return self._lower_bounds

    def get_upper_bounds(self):
        return self._upper_bounds

    def finalize_optimized_parameters(self, parameters):
        return parameters

    def numdiff_get_max_step(self):
        return self._numdiff_step

    def numdiff_get_scaling_factors(self):
        return self._numdiff_scaling_factors

    def numdiff_use_bounds(self):
        return self._numdiff_use_bounds

    def numdiff_parameter_transformation(self):
        return self._numdiff_param_transform

    def numdiff_use_upper_bounds(self):
        return self._numdiff_use_upper_bounds

    def numdiff_use_lower_bounds(self):
        return self._numdiff_use_lower_bounds


class SimpleSampleModel(OptimizeModelInterface, SampleModelInterface):

    def __init__(self, wrapped_optimize_model, ll_per_obs_func, log_prior_function_builder,
                 finalize_proposal_function_builder):
        self._wrapped_optimize_model = wrapped_optimize_model
        self._ll_per_obs_func = ll_per_obs_func
        self._log_prior_function_builder = log_prior_function_builder
        self._finalize_proposal_function_builder = finalize_proposal_function_builder

    def __getattribute__(self, item):
        try:
            value = super(SimpleSampleModel, self).__getattribute__(item)
            if hasattr(SampleModelInterface, item) or hasattr(OptimizeModelInterface, item):
                if inspect.ismethod(value) or inspect.isfunction(value):
                    if not issubclass(get_class_that_defined_method(value), SimpleSampleModel):
                        raise NotImplementedError()
            return value
        except NotImplementedError:
            return getattr(super(SimpleSampleModel, self).__getattribute__('_wrapped_optimize_model'), item)
        except AttributeError:
            return getattr(super(SimpleSampleModel, self).__getattribute__('_wrapped_optimize_model'), item)

    def get_log_likelihood_per_observation_function(self):
        return self._ll_per_obs_func

    def get_log_prior_function(self, address_space_parameter_vector='private'):
        return self._log_prior_function_builder(address_space_parameter_vector)

    def get_finalize_proposal_function(self, address_space_parameter_vector='private'):
        return self._finalize_proposal_function_builder(address_space_parameter_vector)
