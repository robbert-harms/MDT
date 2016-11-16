import collections
import copy

__author__ = 'Robbert Harms'
__date__ = "2016-11-10"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DeferredActionDict(collections.MutableMapping):

    def __init__(self, func, items, memoize=True):
        """Applies the given function on the given items at moment of request.

        On the moment one of the keys of this dict class is requested we apply the given function on the given items
        and return the result of that function. The advantage of this class is that it defers an expensive operation
        until it is needed.

        Args:
            func (function): the callback function to apply on the given items at request, with signature:

                .. code-block: python

                    def callback(key, value)

            items (dict): the items on which we operate
            memoize (boolean): if true we memorize the function output internally. If False we apply the given function
                on every request.
        """
        self._func = func
        self._items = copy.copy(items)
        self._memoize = memoize
        self._memoized = {}

    def __delitem__(self, key):
        del self._items[key]
        if key in self._memoized:
            del self._memoized[key]

    def __getitem__(self, key):
        if not self._memoize:
            return self._func(key, self._items[key])

        if key not in self._memoized:
            self._memoized[key] = self._func(key, self._items[key])
        return self._memoized[key]

    def __contains__(self, key):
        try:
            self._items[key]
        except KeyError:
            return False
        else:
            return True

    def __iter__(self):
        for key in self._items.keys():
            yield key

    def __len__(self):
        return len(self._items)

    def __setitem__(self, key, value):
        self._memoized[key] = value


class DeferredFunctionDict(collections.MutableMapping):

    def __init__(self, items, memoize=True):
        """The items should contain a list of functions that we apply at the moment of request.

        On the moment one of the keys of this dict class is requested we apply the function stored in the items dict
        for that key and return the result of that function.
        The advantage of this class is that it defers an expensive operation until it is needed.

        Args:
            items (dict): the items on which we operate, each value should contain a function with no parameters
                that we run to return the results.
            memoize (boolean): if true we memorize the function output internally. If False we apply the item's function
                on every request.
        """
        self._items = copy.copy(items)
        self._memoize = memoize
        self._memoized = {}

    def __delitem__(self, key):
        del self._items[key]
        if key in self._memoized:
            del self._memoized[key]

    def __getitem__(self, key):
        if not self._memoize:
            return self._items[key]()

        if key not in self._memoized:
            self._memoized[key] = self._items[key]()
        return self._memoized[key]

    def __contains__(self, key):
        try:
            self._items[key]
        except KeyError:
            return False
        else:
            return True

    def __iter__(self):
        for key in self._items.keys():
            yield key

    def __len__(self):
        return len(self._items)

    def __setitem__(self, key, value):
        self._memoized[key] = value


class DeferredActionTuple(collections.Sequence):

    def __init__(self, func, items, memoize=True):
        """Applies the given function on the given items at moment of request.

        On the moment one of the elements is requested we apply the given function on the given items
        and return the result of that function. The advantage of this class is that it defers an expensive operation
        until it is needed.

        Args:
            func (function): the callback function to apply on the given items at request, with signature:

                .. code-block: python

                    def callback(index, value)

            items (list, tuple): the items on which we operate
            memoize (boolean): if true we memorize the function output internally. If False we apply the given function
                on every request.
        """
        self._func = func
        self._items = copy.copy(items)
        self._memoize = memoize
        self._memoized = {}

    def __getitem__(self, index):
        if not self._memoize:
            return self._func(index, self._items[index])

        if index not in self._memoized:
            self._memoized[index] = self._func(index, self._items[index])
        return self._memoized[index]

    def __len__(self):
        return len(self._items)
