VERSION = '0.9.35'

_items = VERSION.split('-')

VERSION_NUMBER_PARTS = tuple(int(i) for i in _items[0].split('.'))

if len(_items) > 1:
    VERSION_STATUS = _items[1]
else:
    VERSION_STATUS = ''

__version__ = VERSION
