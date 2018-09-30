"""Computation caching is a functionality for compartment models.

During optimization and sampling we compute each compartment models for every volume/observation. Some compartments
have heavy computations that can be done only once per model evaluation and which can therefore, in principle,
be cached. The caching functionality in this module enables precisely that, allowing a compartment model to store
some intermediate values in a cache.

During (composite) model evaluation, the composite model will first call the cache update function for each compartment.
Then, when looping over the volumes, the compartments can use the cached computations to evaluate each volume.
"""
from collections import OrderedDict, Mapping

from mot.lib.cl_function import SimpleCLFunction, SimpleCLFunctionParameter

__author__ = 'Robbert Harms'
__date__ = '2018-09-16'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class CacheInfo:

    def __init__(self, fields, cl_code):
        """A storage container for the cache info of an compartment function.

        Args:
            fields (list of CacheData): the list of cache data needed for the cache of this compartment
            cl_code (str): the body of the CL code needed for the cache initialization.
                This will be turned into a function by the compartment model.
        """
        self.fields = fields
        self.cl_code = cl_code


class CacheData:
    pass


class CacheStruct(CacheData):

    def __init__(self, elements, name):
        """A kernel data element for structs.

        Please be aware that structs will always be passed as a pointer to the calling function.

        Args:
            elements (List[CacheData]): the list of cache data elements for this struct
            name (str): the name of this structure
        """
        self._elements = elements
        self._name = name

    def get_type_definitions(self, address_space):
        # other_structs = '\n'.join(element.get_type_definitions(address_space) for element in self._elements)
        return '''
            typedef struct {ctype}_cache{{
                {definitions}
            }} {ctype}_cache;
        '''.format(ctype=self._name,
                   definitions='\n'.join(element.get_struct_declaration(address_space) for element in self._elements))

    def get_struct_declaration(self, address_space):
        return '{0}_cache* {0};'.format(self._name)

    def initialize_variable(self, parent_name, address_space):
        return_str = ''
        for element in self._elements:
            return_str += element.initialize_variable(
                '{}_{}_cache'.format(parent_name, self._name), address_space) + '\n'

        inits = [element.get_struct_initialization('{}_{}_cache'.format(parent_name, self._name))
                 for element in self._elements]

        return return_str + '''
            {name}_cache {parent_name}_{name} = {{ {inits} }};
        '''.format(parent_name=parent_name, name=self._name, inits=', '.join(inits))

    def get_struct_initialization(self, parent_name):
        return '&{}_{}'.format(parent_name, self._name)


class CachePrimitive(CacheData):

    def __init__(self, ctype, name, nmr_elements=1):
        self._ctype = ctype
        self._name = name
        self._nmr_elements = nmr_elements

    def get_type_definitions(self, address_space):
        return ''

    def get_struct_declaration(self, address_space):
        return '{} {}* {};'.format(address_space, self._ctype, self._name)

    def initialize_variable(self, parent_name, address_space):
        if self._nmr_elements > 1:
            return '{} {} {}_{}[{}];'.format(address_space, self._ctype, parent_name, self._name, self._nmr_elements)
        else:
            return '{} {} {}_{};'.format(address_space, self._ctype, parent_name, self._name)

    def get_struct_initialization(self, parent_name):
        if self._nmr_elements > 1:
            return '{}_{}'.format(parent_name, self._name)
        return '&{}_{}'.format(parent_name, self._name)


class DataCacheParameter(SimpleCLFunctionParameter):

    def __init__(self, compartment_name, name):
        """This class provides a subclass for checking instance types.

        Args:
            compartment_name (str): the name of the compartment holding this parameter.
                This parameter will make sure it gets named to the correct caching struct type.
            name (str): the name of this parameter in the function
        """
        super().__init__('{}_cache*'.format(compartment_name), name)
