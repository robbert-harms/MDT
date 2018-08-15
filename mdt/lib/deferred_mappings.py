import collections
import copy

__author__ = 'Robbert Harms'
__date__ = "2016-11-10"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DeferredActionDict(collections.MutableMapping):

    def __init__(self, func, items, cache=True):
        """Applies the given function on the given items at the moment of data request.

        On the moment one of the keys of this dict class is requested we apply the given function on the given items
        and return the result of that function. The advantage of this class is that it defers an expensive operation
        until it is needed.

        Items added to this dictionary after creation are assumed to be final, that is, we won't run the
        function on them.

        Args:
            func (Function): the callback function to apply on the given items at request, with signature:

                .. code-block:: python

                    def callback(key, value)

            items (collections.MutableMapping): the items on which we operate
            cache (boolean): if we want to cache computed results
        """
        self._func = func
        self._items = copy.copy(items)
        self._applied_on_key = {}
        self._cache = cache

    def __delitem__(self, key):
        if key in self._items:
            del self._items[key]
        if key in self._applied_on_key:
            del self._applied_on_key[key]

    def __getitem__(self, key):
        if key not in self._applied_on_key or not self._applied_on_key[key]:
            item = self._func(key, self._items[key])

            if not self._cache:
                return item

            self._items[key] = item
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
        new_one = type(self)(self._func, copy.copy(self._items), cache=self._cache)
        new_one._applied_on_key = copy.copy(self._applied_on_key)
        return new_one


class DeferredFunctionDict(collections.MutableMapping):

    def __init__(self, items, cache=True):
        """The items should contain a list of functions that we apply at the moment of request.

        On the moment one of the keys of this dict class is requested we apply the function stored in the items dict
        for that key and return the result of that function.
        The advantage of this class is that it defers an expensive operation until it is needed.

        Items set to this dictionary are assumed to be final, that is, we won't run the function on them.

        Args:
            items (collections.MutableMapping): the items on which we operate, each value should
                contain a function with no parameters that we run to return the results.
            cache (boolean): if we want to cache computed results
        """
        self._items = copy.copy(items)
        self._applied_on_key = {}
        self._cache = cache

    def __delitem__(self, key):
        if key in self._items:
            del self._items[key]
        if key in self._applied_on_key:
            del self._applied_on_key[key]

    def __getitem__(self, key):
        if key not in self._applied_on_key or not self._applied_on_key[key]:
            item = self._items[key]()

            if not self._cache:
                return item

            self._items[key] = item
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
        new_one = type(self)(copy.copy(self._items), cache=self._cache)
        new_one._applied_on_key = copy.copy(self._applied_on_key)
        return new_one


class DeferredActionTuple(collections.Sequence):

    def __init__(self, func, items, cache=True):
        """Applies the given function on the given items at moment of request.

        On the moment one of the elements is requested we apply the given function on the given items
        and return the result of that function. The advantage of this class is that it defers an expensive operation
        until it is needed.

        Args:
            func (Function): the callback function to apply on the given items at request, with signature:

                .. code-block:: python

                    def callback(index, value)

            items (list, tuple): the items on which we operate
            cache (boolean): if we want to cache computed results
        """
        self._func = func
        self._items = list(copy.copy(items))
        self._applied_on_index = {}
        self._cache = cache

    def __getitem__(self, index):
        if index not in self._applied_on_index or not self._applied_on_index[index]:
            item = self._func(index, self._items[index])

            if not self._cache:
                return item

            self._items[index] = item
            self._applied_on_index[index] = True
        return self._items[index]

    def __len__(self):
        return len(self._items)

    def __copy__(self):
        new_one = type(self)(self._func, copy.copy(self._items), cache=self._cache)
        new_one._applied_on_index = copy.copy(self._applied_on_index)
        return new_one
