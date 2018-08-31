from collections import Mapping, Sequence

__author__ = 'Robbert Harms'
__date__ = "2016-09-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ConversionSpecification:

    def __init__(self):
        """Specifies how the content of an object is to be converted from and to a dictionary."""

    def to_dict(self, obj):
        """Convert the given value to a dictionary.

        Args:
            obj : the value to convert to a dictionary

        Returns:
            dict: the resulting converted dictionary
        """

    def from_dict(self, value):
        """Generate a result value from the given dictionary

        Args:
            value : the dictionary to convert back to a value

        Returns:
            object: the value represented by the dict.
        """


class OptionalConversionDecorator(ConversionSpecification):

    def __init__(self, conversion_specification):
        """Makes the conversion optional by testing against None.

        If the element to convert is None, we will return None as a conversion. If the element to convert is not None
        we will convert it according to the conversion specified.

        This holds for both to- and from- dict.

        Args:
            conversion_specification (ConversionSpecification): the conversion specification to use if the element
                to convert is not None.
        """
        super().__init__()
        self._conversion_specification = conversion_specification

    def to_dict(self, obj):
        if obj is None:
            return None
        return self._conversion_specification.to_dict(obj)

    def from_dict(self, value):
        if value is None:
            return None
        return self._conversion_specification.from_dict(value)


class SimpleClassConversion(ConversionSpecification):

    def __init__(self, class_type, attribute_conversions):
        """Converts a dictionary to and from the specified class.

        Args:
            class_type (type): the type of class to convert to
            attribute_conversions (List[ConversionSpecification]) the list of conversion specification for the
                attributes
        """
        super().__init__()
        self._class_type = class_type
        self._attribute_conversions = attribute_conversions

    def to_dict(self, obj):
        result_dict = {}
        conversion_info = self._attribute_conversions
        for key, converter in conversion_info.items():
            result_dict[key] = converter.to_dict(getattr(obj, key))
        return result_dict

    def from_dict(self, value):
        init_kwargs = {}
        for key, converter in self._attribute_conversions.items():
            if key in value:
                init_kwargs[key] = converter.from_dict(value[key])
        return self._class_type(**init_kwargs)


class ConvertDictElements(ConversionSpecification):

    def __init__(self, conversion_type):
        """Converts all the elements in the value (a dictionary) using the given conversion type."""
        super().__init__()
        self._conversion_type = conversion_type

    def to_dict(self, obj):
        return {key: self._conversion_type.to_dict(v) for key, v in obj.items()}

    def from_dict(self, value):
        return {key: self._conversion_type.from_dict(v) for key, v in value.items()}


class ConvertListElements(ConversionSpecification):

    def __init__(self, conversion_type):
        """Converts all the elements in the value (a list) using the given conversion type."""
        super().__init__()
        self._conversion_type = conversion_type

    def to_dict(self, obj):
        return [self._conversion_type.to_dict(v) for v in obj]

    def from_dict(self, value):
        return [self._conversion_type.from_dict(v) for v in value]


class ConvertDynamicFromModule(ConversionSpecification):

    def __init__(self, module):
        """Performs dynamic lookup by loading the class from the given module.

        This requires that the class we are dynamically loading has a get_conversion_info() class method that
        returns the conversion specification for that class.

        Args:
            module (module): the python module to use for loading the data from dict
        """
        super().__init__()
        self._module = module

    def to_dict(self, obj):
        return [obj.__class__.__name__, obj.get_conversion_info().to_dict(obj)]

    def from_dict(self, value):
        try:
            cls = getattr(self._module, value[0])
        except AttributeError:
            raise ValueError('The given class "{}" could not be found.'.format(value[0]))

        return cls.get_conversion_info().from_dict(value[1])


class SimpleFunctionConversion(ConversionSpecification):

    def __init__(self, conversion_func=None, allow_null=True, set_null_to_value=None):
        """Performs identity conversion between simple types.

        Args:
            conversion_func (Func): if not None we apply the given function before converting to and from the
                dictionary. Can also be a type like ``int``.
            allow_null (bool): if True we allow None during type casting
            set_null_to_value (obj): the value to set null entries to. If this is None and allow_null is False we
                raise an error.
        """
        super().__init__()
        self._conversion_func = conversion_func
        self._allow_none = allow_null
        self._set_null_to_value = set_null_to_value

    def to_dict(self, obj):
        if obj is None:
            if self._allow_none:
                return None
            else:
                if self._set_null_to_value is not None:
                    return self._set_null_to_value
                raise ValueError('The object is supposed to be not None.')
        else:
            if self._conversion_func:
                return self._conversion_func(obj)
        return obj

    def from_dict(self, value):
        if value is None:
            if self._allow_none:
                return None
            else:
                if self._set_null_to_value is not None:
                    return self._set_null_to_value
                raise ValueError('The object is supposed to be not None.')
        else:
            if self._conversion_func:
                return self._conversion_func(value)
        return value


class SimpleDictConversion(SimpleFunctionConversion):

    def __init__(self, conversion_func=None, allow_null=True, set_null_to_value=None):
        """Converts all the objects in the given dict.

        Args:
            conversion_func (Func): if not None we cast the from_dict value to the given type
            allow_null (bool): if True we allow None during type casting
            set_null_to_value (obj): the value to set null entries to. If this is None and allow_null is False we
                raise an error.
        """
        super().__init__(
            conversion_func=SimpleDictConversion._get_conversion_func(conversion_func),
            allow_null=allow_null,
            set_null_to_value=set_null_to_value)

    @staticmethod
    def _get_conversion_func(user_conversion_func):
        """Wraps the desired type into a Dict[user_conversion_func] function."""
        if user_conversion_func is None:
            return None

        def conversion_wrapper(obj):
            if isinstance(obj, Mapping):
                return {key: user_conversion_func(v) for key, v in obj.items()}
            return obj

        return conversion_wrapper


class SimpleListConversion(SimpleFunctionConversion):

    def __init__(self, conversion_func=None, allow_null=True, set_null_to_value=None):
        """Converts all the objects in the given list.

        Args:
            conversion_func (Func): if not None we cast the from_dict value to the given type
            allow_null (bool): if True we allow None during type casting
            set_null_to_value (obj): the value to set null entries to. If this is None and allow_null is False we
                raise an error.
        """
        super().__init__(
            conversion_func=SimpleListConversion._get_conversion_func(conversion_func),
            allow_null=allow_null,
            set_null_to_value=set_null_to_value)

    @staticmethod
    def _get_conversion_func(desired_type):
        """Wraps the desired type into a dict(desired_type) function."""
        if desired_type is None:
            return None

        def conversion_wrapper(obj):
            if isinstance(obj, Sequence):
                return [desired_type(el) for el in obj]
            return obj

        return conversion_wrapper


class StringConversion(SimpleFunctionConversion):

    def __init__(self, allow_null=True):
        super().__init__(str, allow_null=allow_null)


class IntConversion(SimpleFunctionConversion):

    def __init__(self, allow_null=True):
        super().__init__(int, allow_null=allow_null)


class FloatConversion(SimpleFunctionConversion):

    def __init__(self, allow_null=True):
        super().__init__(float, allow_null=allow_null)


class BooleanConversion(SimpleFunctionConversion):

    def __init__(self, allow_null=True):
        super().__init__(bool, allow_null=allow_null)


class WhiteListConversion(ConversionSpecification):

    def __init__(self, white_list, default):
        """Allow only elements from the given white list. If the element is not one of them, revert to the default.

        Args:
            white_list (list of object): list of allowable objects
            default : the default fallback object
        """
        super().__init__()
        self.white_list = white_list
        self.default = default

    def to_dict(self, obj):
        if obj not in self.white_list:
            return self.default
        return obj

    def from_dict(self, value):
        if value not in self.white_list:
            return self.default
        return value
