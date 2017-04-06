from collections import Mapping, Sequence

__author__ = 'Robbert Harms'
__date__ = "2016-09-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ConversionSpecification(object):

    def __init__(self):
        """Specifies how the content of an object is to be converted from and to a dictionary."""

    def to_dict(self, obj):
        """Convert the given value to a dictionary.

        Args:
            obj (object): the value to convert to a dictionary

        Returns:
            dict: the resulting converted dictionary
        """

    def from_dict(self, value):
        """Generate a result value from the given dictionary

        Args:
            value (object): the dictionary to convert back to a value

        Returns:
            object: the value represented by the dict.
        """


class SimpleClassConversion(ConversionSpecification):

    def __init__(self, class_type, attribute_conversions):
        super(SimpleClassConversion, self).__init__()
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
        super(ConvertDictElements, self).__init__()
        self._conversion_type = conversion_type

    def to_dict(self, obj):
        return {key: self._conversion_type.to_dict(v) for key, v in obj.items()}

    def from_dict(self, value):
        return {key: self._conversion_type.from_dict(v) for key, v in value.items()}


class ConvertListElements(ConversionSpecification):

    def __init__(self, conversion_type):
        """Converts all the elements in the value (a list) using the given conversion type."""
        super(ConvertListElements, self).__init__()
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
        super(ConvertDynamicFromModule, self).__init__()
        self._module = module

    def to_dict(self, obj):
        return [obj.__class__.__name__, obj.get_conversion_info().to_dict(obj)]

    def from_dict(self, value):
        try:
            cls = getattr(self._module, value[0])
        except AttributeError:
            raise ValueError('The given class "{}" could not be found.'.format(value[0]))

        return cls.get_conversion_info().from_dict(value[1])


class IdentityConversion(ConversionSpecification):

    def __init__(self, desired_type=None, allow_null=True):
        """Performs identity conversion between simple types.

        Args:
            desired_type (:class:`type`): if not None we cast the from_dict value to the given type
            allow_null (bool): if True we allow None during type casting
        """
        super(IdentityConversion, self).__init__()
        self._desired_type = desired_type
        self._allow_none = allow_null

    def to_dict(self, obj):
        if obj is None:
            if self._allow_none:
                return None
            else:
                raise ValueError('The object is supposed to be not None.')
        else:
            if self._desired_type:
                return self._desired_type(obj)
        return obj

    def from_dict(self, value):
        if value is None:
            if self._allow_none:
                return None
            else:
                raise ValueError('The object is supposed to be not None.')
        else:
            if self._desired_type:
                return self._desired_type(value)
        return value


class SimpleDictConversion(IdentityConversion):

    def __init__(self, desired_type=None, allow_null=True):
        """Converts all the objects in the given dict.

        Args:
            desired_type (:class:`type`): if not None we cast the from_dict value to the given type
            allow_null (bool): if True we allow None during type casting
        """
        super(SimpleDictConversion, self).__init__(
            desired_type=SimpleDictConversion._get_conversion_func(desired_type),
            allow_null=allow_null)

    @staticmethod
    def _get_conversion_func(desired_type):
        """Wraps the desired type into a dict(desired_type) function."""
        if desired_type is None:
            return None

        def conversion_wrapper(obj):
            if isinstance(obj, Mapping):
                return {key: desired_type(v) for key, v in obj.items()}
            return obj

        return conversion_wrapper


class SimpleListConversion(IdentityConversion):

    def __init__(self, desired_type=None, allow_null=True):
        """Converts all the objects in the given list.

        Args:
            desired_type (:class:`type`): if not None we cast the from_dict value to the given type
            allow_null (bool): if True we allow None during type casting
        """
        super(SimpleListConversion, self).__init__(
            desired_type=SimpleListConversion._get_conversion_func(desired_type),
            allow_null=allow_null)

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


class StringConversion(IdentityConversion):

    def __init__(self, allow_null=True):
        super(StringConversion, self).__init__(str, allow_null=allow_null)


class IntConversion(IdentityConversion):

    def __init__(self, allow_null=True):
        super(IntConversion, self).__init__(int, allow_null=allow_null)


class FloatConversion(IdentityConversion):

    def __init__(self, allow_null=True):
        super(FloatConversion, self).__init__(float, allow_null=allow_null)


class BooleanConversion(IdentityConversion):

    def __init__(self, allow_null=True):
        super(BooleanConversion, self).__init__(bool, allow_null=allow_null)


class WhiteListConversion(ConversionSpecification):

    def __init__(self, white_list, default):
        """Allow only elements from the given white list. If the element is not one of them, revert to the default.

        Args:
            white_list (list of object): list of allowable objects
            default (object): the default fallback object
        """
        super(WhiteListConversion, self).__init__()
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
