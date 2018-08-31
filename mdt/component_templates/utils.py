import ast
import inspect
from textwrap import indent, dedent

__author__ = 'Robbert Harms'
__date__ = '2017-07-24'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class TemplateModifier:

    def __init__(self, template):
        """Given a template instance, this class can update the properties of the template and write that to a file.

        This will overwrite the template definition in the source file where the template was originally defined. If you
        desire a different location then first create a new file for the template.

        Args:
            template (ComponentTemplate): the template to update
        """
        self._source = inspect.getsource(template)
        self._filename = inspect.getfile(template)

        source_lines = inspect.getsourcelines(template)
        self._template_start = source_lines[1] - 1
        self._template_end = len(source_lines[0]) + self._template_start

    def update(self, property_name, source_code_str):
        """Update the given property with the given source code.

        This does not write the results to file immediately, rather, this updates an internal buffer with the
        updated source code. To write to file use :meth:`write_to_file`.

        Args:
            property_name (str): the property (attribute or function) to update
            source_code_str (str): the updated source code for the property
        """
        try:
            start, end = self._fine_property_definition(property_name)
            source_lines = self._source.split('\n')
            new_source = '\n'.join(source_lines[:start])
            new_source += '\n' + indent(dedent(source_code_str.strip()), '\t') + '\n'
            new_source += '\n'.join(source_lines[end:])
        except ValueError:
            new_source = self._source + '\n' + indent(dedent(source_code_str.strip()), '\t') + '\n'
        self._source = new_source.rstrip('\n').replace('\t', '    ')

    def get_source(self):
        """Return the current source code buffer.

        Returns:
            str: the updated source code
        """
        return self._source

    def write_to_file(self):
        """Write the updated source code to file.

        This can be called repetitively after calling :meth:`update` since this method will keep track of the current
        position in the file.
        """
        with open(self._filename, 'r') as f:
            lines = [line.rstrip('\n') for line in f]

        start_lines = lines[:self._template_start]
        new_source_lines = self._source.split('\n')
        end_lines = lines[self._template_end:]

        lines = start_lines + new_source_lines + end_lines

        with open(self._filename, 'w') as f:
            f.writelines('{}\n'.format(l) for l in lines)

        self._template_start = len(start_lines)
        self._template_end = len(start_lines) + len(new_source_lines)

    def _fine_property_definition(self, property_name):
        """Find the lines in the source code that contain this property's name and definition.

        This function can find both attribute assignments as well as methods/functions.

        Args:
            property_name (str): the name of the property to look up in the template definition

        Returns:
            tuple: line numbers for the start and end of the attribute definition
        """
        for node in ast.walk(ast.parse(self._source)):
            if isinstance(node, ast.Assign) and node.targets[0].id == property_name:
                return node.targets[0].lineno - 1, self._get_node_line_end(node)
            elif isinstance(node, ast.FunctionDef) and node.name == property_name:
                return node.lineno - 1, self._get_node_line_end(node)
        raise ValueError('The requested node could not be found.')

    def _get_node_line_end(self, node):
        """Get the last line of the given node.

        This function can recurse if the given node is a complex node (like a FunctionDef node).

        Args:
            node (ast.Node): a node of the AST

        Returns:
            int: the last line of the statements in the given node
        """
        if isinstance(node, ast.Assign):
            return node.value.lineno
        elif isinstance(node, ast.FunctionDef):
            return self._get_node_line_end(node.body[-1])
