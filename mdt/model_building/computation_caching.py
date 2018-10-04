"""Computation caching is a functionality for compartment models.

During optimization and sampling we compute each compartment models for every volume/observation. Some compartments
have heavy computations that can be done only once per model evaluation and which can therefore, in principle,
be cached. The caching functionality in this module enables precisely that, allowing a compartment model to store
some intermediate values in a cache.

During (composite) model evaluation, the composite model will first call the cache update function for each compartment.
Then, when looping over the volumes, the compartments can use the cached computations to evaluate each volume.
"""
from mot.lib.cl_function import SimpleCLFunctionParameter

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
    """The cache data elements specify the data types and variable declaration."""

    def get_type_definitions(self, address_space):
        """Get the CL that defines this cache data type.

        This can be used to define structures, or other kind of typedefs needed for this cache data element.
        These type definitions should contain include guards to prevent double definitions.

        Returns:
            str: a piece of CL code defining the type of this cache element.
        """
        raise NotImplementedError()

    def get_type_name(self):
        """Get the c-type of this cache data element.

        Returns:
            str: the c-type of this cache element
        """
        raise NotImplementedError()

    def get_variable_name(self, parent_name=''):
        """Get the name of the variable used in the struct variable declaration.

        Args:
            parent_name (str): the parent compartment name, used for nested compartments. This does generally not need
                to be set by users of cache data objects.

        Returns:
            str: the name of the variable holding the cache declaration inside a CL function
        """
        raise NotImplementedError()

    def get_field_name(self):
        """Get the field name of this cache data element when used in a struct.

        Returns:
            str: the name of this data element inside a struct
        """
        raise NotImplementedError()

    def get_variable_declaration(self, address_space, parent_name=''):
        """Declare all variables necessary for holding this cache data information.

        This is supposed to be used inside a CL function for declaring this cache data element. For a structure,
        this will first declare all the elements of the structure and then the structure itself.

        Args:
            address_space (str): the address space for the primitives
            parent_name (str): for nested cache data elements, the name of the parent element. Typically used
                for nested compartments.

        Returns:
            str: declaration code for this cache data element
        """
        raise NotImplementedError()

    def get_struct_initialization(self, parent_name=''):
        """Get the code for initializing this cache data element inside a structure.

        This is for private use of cache structures.

        Args:
            parent_name (str): for nested cache data elements, the name of the parent element. Typically used
                for nested compartments.

        Returns:
            str: the code for initializing the declared variable inside a struct.
        """
        raise NotImplementedError()

    def get_struct_field_declaration(self, address_space):
        """Get the declaration of this cache element inside a structure.

        Args:
            address_space (str): the desired address space of the variable pointer.

        Returns:
            str: the code for struct field declaration of this element.
        """
        raise NotImplementedError()


class CacheStruct(CacheData):

    def __init__(self, ctype, elements, field_name, variable_name):
        """A kernel data element for structs.

        Please be aware that structs will always be passed as a pointer to the calling function.

        Args:
            ctype (str): the ctype for this cache, typically set to "<compartment_name>_cache"
            elements (List[CacheData]): the list of cache data elements for this struct
            field_name (str): the field name of this element in the cache struct
            variable_name (str): the variable name for the definition of this structure
        """
        self._ctype = ctype
        self._elements = elements
        self._field_name = field_name
        self._variable_name = variable_name

    def get_type_definitions(self, address_space):
        other_structs = '\n'.join(element.get_type_definitions(address_space) for element in self._elements)
        return other_structs + '''
            #ifndef CACHESTRUCT_{ctype}
            #define CACHESTRUCT_{ctype}
            
            typedef struct {ctype}{{
                {definitions}
            }} {ctype};
                        
            #endif /* CACHESTRUCT_{ctype} */
        '''.format(ctype=self._ctype,
                   definitions='\n'.join(element.get_struct_field_declaration(address_space)
                                         for element in self._elements))

    def get_type_name(self):
        return self._ctype

    def get_variable_name(self, parent_name=''):
        return '{}_{}'.format(parent_name, self._variable_name)

    def get_field_name(self):
        return self._field_name

    def get_variable_declaration(self, address_space, parent_name=''):
        return_str = ''
        for element in self._elements:
            return_str += element.get_variable_declaration(address_space, self.get_variable_name(parent_name))

        inits = [element.get_struct_initialization(self.get_variable_name(parent_name)) for element in self._elements]

        return return_str + '''
            {ctype} {var_name} = {{ {inits} }};
        '''.format(ctype=self._ctype, var_name=self.get_variable_name(parent_name), inits=', '.join(inits))

    def get_struct_initialization(self, parent_name=''):
        return '&{}'.format(self.get_variable_name(parent_name))

    def get_struct_field_declaration(self, address_space):
        return '{}* {};'.format(self._ctype, self._field_name)


class CachePrimitive(CacheData):

    def __init__(self, ctype, name, nmr_elements=1):
        self._ctype = ctype
        self._name = name
        self._nmr_elements = nmr_elements

    def get_type_definitions(self, address_space):
        return ''

    def get_type_name(self):
        return self._ctype

    def get_variable_name(self, parent_name=''):
        return '{}_{}'.format(parent_name, self._name)

    def get_field_name(self):
        return self._name

    def get_variable_declaration(self, address_space, parent_name=''):
        if self._nmr_elements > 1:
            return '{addr} {ctype} {name}[{nmr_elements}];'.format(
                addr=address_space, ctype=self._ctype, name=self.get_variable_name(parent_name),
                nmr_elements=self._nmr_elements)
        else:
            return '{addr} {ctype} {name};'.format(addr=address_space, ctype=self._ctype,
                                                   name=self.get_variable_name(parent_name))

    def get_struct_initialization(self, parent_name=''):
        if self._nmr_elements > 1:
            return self.get_variable_name(parent_name)
        return '&{}_{}'.format(parent_name, self._name)

    def get_struct_field_declaration(self, address_space):
        return '{} {}* {};'.format(address_space, self._ctype, self._name)


class DataCacheParameter(SimpleCLFunctionParameter):

    def __init__(self, compartment_name, name):
        """This class provides a subclass for checking instance types.

        Args:
            compartment_name (str): the name of the compartment holding this parameter.
                This parameter will make sure it gets named to the correct caching struct type.
            name (str): the name of this parameter in the function
        """
        super().__init__('{}_cache*'.format(compartment_name), name)
