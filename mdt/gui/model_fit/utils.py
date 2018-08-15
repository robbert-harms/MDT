import glob
import os
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
        super().__init__(*args, **kwargs)

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


def results_preselection_names(data):
    """Generate a list of useful map names to display.

    This is primarily to be used as argument to the config option ``maps_to_show`` in the function :func:`view_maps`.

    Args:
        data (str or dict or list of str): either a directory or a dictionary of results or a list of map names.

    Returns:
        list of str: the list of useful/filtered map names.
    """
    keys = []
    if isinstance(data, str):
        for extension in ('.nii', '.nii.gz'):
            for f in glob.glob(os.path.join(data, '*' + extension)):
                keys.append(os.path.basename(f)[0:-len(extension)])
    elif isinstance(data, dict):
        keys = data.keys()
    else:
        keys = data

    filter_match = ('.vec', '.d', '.sigma', '.theta', '.phi', 'AIC', 'Errors', 'Errors', '.eigen_ranking',
                    'SignalEstimates', 'UsedMask', 'BIC')
    return list(sorted(filter(lambda v: all(m not in v for m in filter_match), keys)))

