import six
from mdt.components_loader import CompartmentModelsLoader
from .CompositeModelExpression import CompositeModelExpressionSemantics, CompositeModelExpressionParser


class Semantics(CompositeModelExpressionSemantics):

    def __init__(self):
        super(Semantics, self).__init__()
        self._compartments_loader = CompartmentModelsLoader()

    def expr(self, ast):
        if not isinstance(ast, list):
            return ast
        if isinstance(ast, list):
            return ast[0], ast[2], ast[1]
        return ast

    def term(self, ast):
        if not isinstance(ast, list):
            return ast
        if isinstance(ast, list):
            return ast[0], ast[2], ast[1]
        return ast

    def factor(self, ast):
        if isinstance(ast, list):
            return ast[1]
        return ast

    def model(self, ast):
        if isinstance(ast, six.string_types):
            return self._compartments_loader.load(ast)
        else:
            return self._compartments_loader.load(ast[0], ast[2])


def parse(model_expression):
    """Parse the given model expression into a suitable model tree.

    Args:
        model_expression (str): the model expression string. Example:

        .. code-block:: none

            S0 * ( (Weight(Wball) * Ball) +
                   (Weight(Wstick) * Stick ) )

        If the model name is followed by parenthesis the string in parenthesis will represent the model's nickname.

    Returns:
        :class:`list`: the compartment model tree for use in composite models.
    """
    parser = CompositeModelExpressionParser(parseinfo=False)
    return parser.parse(model_expression, rule_name='result', semantics=Semantics())
