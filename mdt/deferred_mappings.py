import collections
import copy

__author__ = 'Robbert Harms'
__date__ = "2016-11-10"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DeferredActionDict(collections.MutableMapping):

    def __init__(self, func, items):
        """Applies the given function on the given items at the moment of data request.

        On the moment one of the keys of this dict class is requested we apply the given function on the given items
        and return the result of that function. The advantage of this class is that it defers an expensive operation
        until it is needed.

        Args:
            func (function): the callback function to apply on the given items at request, with signature:

                .. code-block: python

                    def callback(key, value)

            items (dict): the items on which we operate
        """
        self._func = func
        self._items = copy.copy(items)
        self._applied_on_key = {}

    def __delitem__(self, key):
        if key in self._items:
            del self._items[key]
        if key in self._applied_on_key:
            del self._applied_on_key[key]

    def __getitem__(self, key):
        if key not in self._applied_on_key or not self._applied_on_key[key]:
            self._func(key, self._items[key])
            self._items[key] = self._func(key, self._items[key])
            self._applied_on_key[key] = True
        return self._items[key]

    def __contains__(self, key):
        return key in self._items

    def __iter__(self):
        for key in self._items.keys():
            yield key

    def __len__(self):
        return len(self._items)

    def __setitem__(self, key, value):
        self._items[key] = value
        self._applied_on_key[key] = True

    def __copy__(self):
        new_one = type(self)(self._func, copy.copy(self._items))
        new_one._applied_on_key = copy.copy(self._applied_on_key)
        return new_one


class DeferredFunctionDict(collections.MutableMapping):

    def __init__(self, items):
        """The items should contain a list of functions that we apply at the moment of request.

        On the moment one of the keys of this dict class is requested we apply the function stored in the items dict
        for that key and return the result of that function.
        The advantage of this class is that it defers an expensive operation until it is needed.

        Args:
            items (dict): the items on which we operate, each value should contain a function with no parameters
                that we run to return the results.
        """
        self._items = copy.copy(items)
        self._applied_on_key = {}

    def __delitem__(self, key):
        if key in self._items:
            del self._items[key]
        if key in self._applied_on_key:
            del self._applied_on_key[key]

    def __getitem__(self, key):
        if key not in self._applied_on_key or not self._applied_on_key[key]:
            self._items[key] = self._items[key]()
            self._applied_on_key[key] = True
        return self._items[key]

    def __contains__(self, key):
        return key in self._items

    def __iter__(self):
        for key in self._items.keys():
            yield key

    def __len__(self):
        return len(self._items)

    def __setitem__(self, key, value):
        self._items[key] = value
        self._applied_on_key[key] = True

    def __copy__(self):
        new_one = type(self)(copy.copy(self._items))
        new_one._applied_on_key = copy.copy(self._applied_on_key)
        return new_one


class DeferredActionTuple(collections.Sequence):

    def __init__(self, func, items):
        """Applies the given function on the given items at moment of request.

        On the moment one of the elements is requested we apply the given function on the given items
        and return the result of that function. The advantage of this class is that it defers an expensive operation
        until it is needed.

        Args:
            func (function): the callback function to apply on the given items at request, with signature:

                .. code-block: python

                    def callback(index, value)

            items (list, tuple): the items on which we operate
        """
        self._func = func
        self._items = copy.copy(items)
        self._applied_on_index = {}

    def __getitem__(self, index):
        if index not in self._applied_on_index or not self._applied_on_index[index]:
            self._items[index] = self._func(index, self._items[index])
            self._applied_on_index[index] = True
        return self._items[index]

    def __len__(self):
        return len(self._items)

    def __copy__(self):
        new_one = type(self)(self._func, copy.copy(self._items))
        new_one._applied_on_index = copy.copy(self._applied_on_index)
        return new_one
