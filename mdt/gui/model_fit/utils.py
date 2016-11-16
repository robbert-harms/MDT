from PyQt5.QtCore import QObject, pyqtSignal

from mdt.gui.utils import UpdateDescriptor

__author__ = 'Robbert Harms'
__date__ = "2015-08-20"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SharedState(QObject):

    state_updated_signal = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        """The shared state for the model fitting GUI

        Attributes:
            base_dir (str): the base dir for all file opening operations
            dimension_index (int): the dimension index used in various operations
            slice_index (int): the slice index used in various operations
        """
        super(SharedState, self).__init__(*args, **kwargs)

        shared_attributes = {'base_dir': None,
                             'dimension_index': 0,
                             'slice_index': 0,
                             'output_folder': None}

        for key, value in shared_attributes.items():
            def get_attribute_setter(attribute_key):
                def setter(value):
                    setattr(self, attribute_key, value)
                return setter

            setattr(self, '_' + key, value)
            setattr(SharedState, key, UpdateDescriptor(key))
            setattr(self, 'set_' + key, get_attribute_setter(key))
