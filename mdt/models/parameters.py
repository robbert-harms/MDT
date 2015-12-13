__author__ = 'Robbert Harms'
__date__ = "2015-12-12"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ParameterBuilder(object):
#     """The model builder to inherit from.
#
#     One can use this to create models in a declarative style. This works because in the constructor we use deepcopy
#     to copy all the relevant material before creating a new instance of the class.
#
#     Class attributes:
#         name (str): the name of the model
#         in_vivo_suitable (boolean): flag indicating if the model is suitable for in vivo data
#         ex_vivo_suitable (boolean): flag indicating if the model is suitable for ex vivo data
#         description (str): model description
#         post_optimization_modifiers (list): a list of modification callbacks for use after optimization. Example:
#             post_optimization_modifiers = [('SNIF', lambda d: 1 - d['Wcsf.w']),
#                                            ...]
#         dependencies (list): the dependencies between model parameters. Example:
#             dependencies = [('Noddi_EC.kappa', SimpleAssignment('Noddi_IC.kappa')),
#                             ...]
#         model_listing (list): the abstract model tree as a list. If this is defined we do not use the model_expression
#         model_expression (str): the model expression. For the syntax see
#             mdt.models.model_expression_parsers.SingleModel.ebnf
#         evaluation_model (EvaluationModel): the evaluation model to use during optimization
#         signal_noise_model (SignalNoiseModel): optional signal noise decorator
#         inits (dict): indicating the initialization values for the parameters. Example:
#             inits = {'Stick.theta: pi}
#         fixes (dict): indicating the constant value for the given parameters. Example:
#             fixes = {'Ball.d': 3.0e-9}
#         upper_bounds (dict): indicating the upper bounds for the given parameters. Example:
#             upper_bounds = {'Stick.theta': pi}
#         lower_bounds (dict): indicating the lower bounds for the given parameters. Example:
#             lower_bounds = {'Stick.theta': 0}
#     """
    name = ''
    description = ''
    data_type = None
    data_type_str = None
    model = None


#     post_optimization_modifiers = ()
#     dependencies = ()
#     model_listing = None
#     model_expression = None
#     evaluation_model = GaussianEvaluationModel().fix('sigma', 1)
#     signal_noise_model = None
#     inits = {}
#     fixes = {}
#     upper_bounds = {}
#     lower_bounds = {}
#
    def __init__(self):
        super(ParameterBuilder, self).__init__()
#
#         self.add_parameter_dependencies(deepcopy(self._get_dependencies()))
#         self.add_post_optimization_modifiers(deepcopy(self._get_post_optimization_modifiers()))
#
#         self._inits = self.inits
#         self._fixes = self.fixes
#         self._lower_bounds = self.lower_bounds
#         self._upper_bounds = self.upper_bounds
#
#         for full_param_name, value in self._inits.items():
#             self.init(full_param_name, value)
#
#         for full_param_name, value in self._fixes.items():
#             self.fix(full_param_name, value)
#
#         for full_param_name, value in self._lower_bounds.items():
#             self.set_lower_bound(full_param_name, value)
#
#         for full_param_name, value in self._upper_bounds.items():
#             self.set_upper_bound(full_param_name, value)
#
    @classmethod
    def meta_info(cls):
        return {'name': cls.name,
                'description': cls.description}
#
#     @classmethod
#     def _get_model_listing(cls):
#         if cls.model_listing is not None:
#             return cls.model_listing
#         return parse(cls.model_expression)
#
#     @classmethod
#     def _get_dependencies(cls):
#         return cls.dependencies
#
#     @classmethod
#     def _get_post_optimization_modifiers(cls):
#         return cls.post_optimization_modifiers
