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
            dict (object): the dictionary to convert back to a value

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

    def __init__(self, desired_type=None, ignore_none=True):
        """Performs identity conversion between simple types.

        Args:
            desired_type (type): if not None we cast the from_dict value to the given type
            ignore_none (bool): if True we ignore None during type casting
        """
        super(IdentityConversion, self).__init__()
        self._desired_type = desired_type
        self._ignore_none = ignore_none

    def to_dict(self, obj):
        if obj is None and self._ignore_none:
            return obj
        if self._desired_type:
            return self._desired_type(obj)
        return obj

    def from_dict(self, value):
        if value is None and self._ignore_none:
            return value
        if self._desired_type:
            return self._desired_type(value)
        return value


class StringConversion(IdentityConversion):

    def __init__(self):
        super(StringConversion, self).__init__(str)


class SimpleDictConversion(IdentityConversion):
    pass


class IntConversion(IdentityConversion):

    def __init__(self):
        super(IntConversion, self).__init__(int)


class FloatConversion(IdentityConversion):

    def __init__(self):
        super(FloatConversion, self).__init__(float)


class SimpleListConversion(IdentityConversion):
    pass


class BooleanConversion(IdentityConversion):

    def __init__(self):
        super(BooleanConversion, self).__init__(bool)
