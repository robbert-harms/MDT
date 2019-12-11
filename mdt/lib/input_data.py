import logging
import numbers
import numpy as np
from mdt.lib.exceptions import NoiseStdEstimationNotPossible
from mdt.utils import is_scalar, create_roi, estimate_noise_std, load_nifti, restore_volumes, load_protocol, load_brain_mask
from mot.lib.utils import all_elements_equal, get_single_value

__author__ = 'Robbert Harms'
__date__ = '2019-12-10'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class MRIInputData:
    """A container for the input data for optimization/sample models."""

    def has_input_data(self, parameter_name):
        """Check if input data for the given parameter is defined in the model.

        Args:
             parameter_name (str): the name of the parameter for which we want to get input data

        Returns:
            boolean: true if there is input data defined for the given parameter, false otherwise
        """
        raise NotImplementedError()

    def get_input_data(self, parameter_name):
        """Get the input data for the given parameter.

        This may compress the input data whenever possible. For example, when all the datapoints are equal it
        may return a single scalar instead of the vector.

        Args:
             parameter_name (str): the name of the parameter for which we want to get input data

        Returns:
            float, ndarray or None: either a scalar, a vector or a matrix with values for the given parameter.

        Raises:
            ValueError: if no suitable value can be found.
        """
        raise NotImplementedError()

    def get_kernel_data(self, parameter_name):
        """Get the input data for the given parameter as a `mot.lib.kernel_data.KernelData` object.

        This may compress the input data whenever possible. For example, when all the datapoints are equal it
        may return a single scalar instead of the vector.

        Args:
            parameter_name (str): the name of the parameter for which we want to get the kernel data

        Returns:
            mot.lib.kernel_data.KernelData: the kernel data object for this parameter
        """
        raise NotImplementedError()
        # todo add

    @property
    def nmr_voxels(self):
        """Get the number of voxels present in this input data.

        Returns:
            int: the number of voxels we have data for
        """
        raise NotImplementedError()

    @property
    def nmr_observations(self):
        """Get the number of observations/data points per voxels.

        The minimum is one observation per problem.

        Returns:
            int: the number of instances per problem (aka data points)
        """
        raise NotImplementedError()

    @property
    def observations(self):
        """Return the observations stored in this input data container.

        Returns:
            ndarray: The list of observed instances per volumes. Should be a (n, d) matrix of type float with for
                n voxels and d volumes the measured signal.
        """
        raise NotImplementedError()

    @property
    def noise_std(self):
        """The noise standard deviation we will use during model evaluation.

        During optimization or sample the model will be evaluated against the observations using a
        likelihood function. Most of these likelihood functions need a standard deviation representing the noise
        in the data.

        Returns:
            number of ndarray: either a scalar or a 2d matrix with one value per problem instance.
        """
        raise NotImplementedError()

    @property
    def protocol(self):
        """Return the protocol data stored in this input data container.

        Returns:
            mdt.protocol.Protocol: The protocol data information mapping.
        """
        raise NotImplementedError()

    @property
    def extra_protocol(self):
        """Get the extra protocol options stored in this input data container.

        This should return the extra protocol options as ROI values only.

        Here one may additionally specify values to be used for the protocol parameters. These additional values can
        be scalars, vectors and/or volumes. This in contrast to the ``protocol`` which only contains
        scalars and vectors. Items specified here will overwrite items from the protocol in the case of
        duplicated names. This parameter can for example be used to specify gradient volumes, instead of a
        gradient in the protocol, for example by specifying::

            extra_protocol = {'g': np.array(...)}

        Per element, the input can be a scalar, a vector, an array, or a filename. If a filename is given
        we will try to interpret it again as a scalar, vector or array.

        Returns:
            dict: additional protocol items.
        """
        raise NotImplementedError()

    @property
    def signal4d(self):
        """Return the 4d volume with on the first three axis the voxel coordinates and on the last axis the volumes.

        Returns:
            ndarray: a 4d numpy array with all the volumes
        """
        raise NotImplementedError()

    @property
    def nifti_header(self):
        """The header of the nifti file to use for writing the results.

        Returns:
            nibabel nifti header
        """
        raise NotImplementedError()

    @property
    def mask(self):
        """Return the mask in use.

        Returns:
            ndarray: the numpy mask array
        """
        raise NotImplementedError()

    @property
    def gradient_deviations(self):
        """Get the gradient deviations for each voxel.

        This should either return a (n, 3, 3) matrix or a (n, m, 3, 3) matrix, where n is the number of voxels and m is
        the number of volumes.

        Returns:
            None or ndarray: the gradient deviations to use during model fitting. If not applicable, return None.
        """
        return None

    @property
    def volume_weights(self):
        """Get the volume weights per voxel.

        These weights are used during model fitting to weigh the objective function values per observation.

        Returns:
            ndarray: The list of observed instances per volumes. Should be a (n, d) matrix of type float16 (half type)
                with for n voxels and d volumes a weight in [0, 1].
        """
        raise NotImplementedError()

    def get_subset(self, volumes_to_keep=None, volumes_to_remove=None):
        """Create a copy of this input data where we only keep a subset of the volumes.

        This creates a a new input data object with a subset of the protocol and the DWI volume, keeping only
        those specified.

        One can either specify a list with volumes to keep or a list with volumes to remove (and we will keep the rest).
        At least one and at most one list must be specified.

        Args:
            volumes_to_keep (list): the list with volumes we would like to keep.
            volumes_to_remove (list): the list with volumes we would like to remove (keeping the others).

        Returns:
            MRIInputData: the new input data
        """
        raise NotImplementedError()


class SimpleMRIInputData(MRIInputData):

    def __init__(self, protocol, signal4d, mask, nifti_header, extra_protocol=None, gradient_deviations=None,
                 noise_std=None, volume_weights=None):
        """An implementation of the input data for diffusion MRI models.

        Args:
            protocol (Protocol): The protocol object used as input data to the model
            signal4d (ndarray): The DWI data (4d matrix)
            mask (ndarray): The mask used to create the observations list
            nifti_header (nifti header): The header of the nifti file to use for writing the results.
            extra_protocol (Dict[str, val]): additional protocol items. Here one may additionally specify values to be
                used for the protocol parameters. These additional values can be scalars, vectors and/or volumes.
                This in contrast to the ``protocol`` which only contains scalars and vectors. Items specified here will
                overwrite items from the protocol in the case of duplicated names. This parameter can for example be
                used to specify gradient volumes, instead of a gradient in the protocol, for example by specifying::

                    extra_protocol = {'g': np.array(...)}

                Per element, the input can be a scalar, a vector, an array, or a filename. If a filename is given
                we will try to interpret it again as a scalar, vector or array.
            gradient_deviations (ndarray): a gradient deviations matrix. The matrix can be provided in multiple formats:

                - an (x, y, z, 9) matrix with per voxel 9 values that constitute the gradient non-linearities
                    according to the HCP guidelines. (see
                    ``www.humanconnectome.org/storage/app/media/documentation/data_release/Q1_Release_Appendix_II.pdf``)
                    If given in this format, we will automatically add the identity matrix to it, as specified by the
                    HCP guidelines.
                - an (x, y, z, 3, 3) matrix with per voxel the deformation matrix. This will be used as given (i.e. no
                    identity matrix will be added to it like in the HCP format).
                - an (x, y, z, m, 3, 3) matrix with per voxel and per volume a deformation matrix. This will be used as
                    given.

            noise_std (number or ndarray): either None for automatic detection,
                or a scalar, or an 3d matrix with one value per voxel.

            volume_weights (ndarray): if given, a float matrix of the same size as the volume with per voxel and volume
                a weight in [0, 1]. If set, these weights are used during model fitting to weigh the objective function
                values per observation.
        """
        self._logger = logging.getLogger(__name__)
        self._signal4d = signal4d
        self._nifti_header = nifti_header
        self._mask = mask
        self._protocol = protocol
        self._observation_list = None
        self._extra_protocol = self._preload_extra_protocol_items(extra_protocol)
        self._noise_std = noise_std

        self._gradient_deviations = gradient_deviations
        self._gradient_deviations_list = None

        self._volume_weights = volume_weights
        self._volume_weights_list = None

        if protocol.length != 0:
            self._nmr_observations = protocol.length
        else:
            self._nmr_observations = signal4d.shape[3]

        if protocol.length != 0 and signal4d is not None and \
                signal4d.shape[3] != 0 and protocol.length != signal4d.shape[3]:
            raise ValueError('Length of the protocol ({}) does not equal the number of volumes ({}).'.format(
                protocol.length, signal4d.shape[3]))

        if self._volume_weights is not None and self._volume_weights.shape != self._signal4d.shape:
            raise ValueError('The dimensions of the volume weights does not match the dimensions of the signal4d.')

    def has_input_data(self, parameter_name):
        try:
            self.get_input_data(parameter_name)
            return True
        except ValueError:
            return False

    def get_input_data(self, parameter_name):
        if parameter_name in self._extra_protocol:
            value = np.array(self._extra_protocol[parameter_name], copy=False)

            if all_elements_equal(value):
                return get_single_value(value)

            if value.ndim >= 3:
                value = create_roi(value, self.mask)
            return value

        if parameter_name in self._protocol:
            if all_elements_equal(self._protocol[parameter_name]):
                return get_single_value(self._protocol[parameter_name])
            return self._protocol[parameter_name]

        raise ValueError('No input data could be find for the parameter "{}".'.format(parameter_name))

    def copy_with_updates(self, **updates):
        """Create a copy of this input data, while setting some of the arguments to new values.

        Args:
            updates (kwargs): with constructor names.
        """
        arg_names = ['protocol', 'signal4d', 'mask', 'nifti_header']
        args = [self._protocol, self.signal4d, self._mask, self.nifti_header]
        kwargs = dict(extra_protocol=self._extra_protocol,
                      gradient_deviations=self._gradient_deviations,
                      noise_std=self._noise_std)

        for ind, arg_name in enumerate(arg_names):
            args[ind] = updates.get(arg_name, args[ind])

        for kwarg_name in kwargs.keys():
            kwargs[kwarg_name] = updates.get(kwarg_name, kwargs[kwarg_name])

        return self.__class__(*args, **kwargs)

    def get_subset(self, volumes_to_keep=None, volumes_to_remove=None):
        if (volumes_to_keep is not None) and (volumes_to_remove is not None):
            raise ValueError('You can not specify both the list with volumes to keep and volumes to remove. Choose one.')
        if (volumes_to_keep is None) and (volumes_to_remove is None):
            raise ValueError('Please specify either the list with volumes to keep or the list with volumes to remove.')

        if volumes_to_keep is not None:
            if is_scalar(volumes_to_keep):
                volumes_to_keep = [volumes_to_keep]
        elif volumes_to_remove is not None:
            if is_scalar(volumes_to_remove):
                volumes_to_remove = [volumes_to_remove]

            volumes_to_keep = list(range(self.nmr_observations))
            volumes_to_keep = [ind for ind in volumes_to_keep if ind not in volumes_to_remove]

        new_protocol = self.protocol
        if self.protocol is not None:
            new_protocol = self.protocol.get_new_protocol_with_indices(volumes_to_keep)

        new_dwi_volume = self.signal4d
        if self.signal4d is not None:
            new_dwi_volume = self.signal4d[..., volumes_to_keep]

        new_volume_weights = self._volume_weights
        if self._volume_weights is not None:
            new_volume_weights = new_volume_weights[..., volumes_to_keep]

        new_gradient_deviations = self._gradient_deviations
        if self._gradient_deviations is not None:
            if self._gradient_deviations.ndim > 4 and self._gradient_deviations.shape[3] == self.protocol.length:
                if self._gradient_deviations.ndim == 5:
                    new_gradient_deviations = self._gradient_deviations[..., volumes_to_keep, :]
                else:
                    new_gradient_deviations = self._gradient_deviations[..., volumes_to_keep, :, :]

        new_extra_protocol = {}
        for key, value in self._extra_protocol.items():
            value = np.array(value, copy=False)
            if value.ndim > 3 and value.shape[3] == self.protocol.length:
                value = value[:, :, :, volumes_to_keep, ...]
            new_extra_protocol[key] = value

        return self.copy_with_updates(protocol=new_protocol, signal4d=new_dwi_volume,
                                      gradient_deviations=new_gradient_deviations,
                                      volume_weights=new_volume_weights, extra_protocol=new_extra_protocol)

    @property
    def nmr_voxels(self):
        return self.observations.shape[0]

    @property
    def nmr_observations(self):
        return self._nmr_observations

    @property
    def signal4d(self):
        return self._signal4d

    @property
    def nifti_header(self):
        return self._nifti_header

    @property
    def gradient_deviations(self):
        if self._gradient_deviations is None:
            return None
        if self._gradient_deviations_list is None:
            grad_dev = create_roi(self._gradient_deviations, self.mask)

            if grad_dev.shape[-1] == 9:  # HCP WUMINN format, Fortran major. Also adds the identity matrix as specified.
                grad_dev = np.reshape(grad_dev, (-1, 3, 3), order='F') + np.eye(3)

            self._gradient_deviations_list = grad_dev

        return self._gradient_deviations_list

    @property
    def volume_weights(self):
        if self._volume_weights is None:
            return None
        if self._volume_weights_list is None:
            self._volume_weights_list = create_roi(self._volume_weights, self.mask).astype(np.float16)
        return self._volume_weights_list

    @property
    def protocol(self):
        return self._protocol

    @property
    def extra_protocol(self):
        return_values = {}
        for key, value in self._extra_protocol.items():
            value = np.array(value, copy=False)
            if len(value.shape) < 3:
                return_values[key] = value
            return_values[key] = create_roi(value, self.mask)
        return return_values

    @property
    def observations(self):
        if self._observation_list is None:
            self._observation_list = create_roi(self.signal4d, self._mask)

            signal_max = np.max(self._observation_list)
            if signal_max < 10:
                logger = logging.getLogger(__name__)
                logger.warning(
                    'Maximum signal intensity is quite low ({}), analysis results are sometimes improved with '
                    'a higher signal intensity (for example, scale input data by 1e5).'.format(signal_max))

        return self._observation_list

    @property
    def mask(self):
        """Return the mask in use

        Returns:
            np.array: the numpy mask array
        """
        return self._mask

    @property
    def noise_std(self):
        def _compute_noise_std():
            if self._noise_std is None:
                try:
                    return estimate_noise_std(self)
                except NoiseStdEstimationNotPossible:
                    self._logger.warning('Failed to obtain a noise std for this subject. '
                                         'We will continue with an std of 1.')
                    return 1

            if isinstance(self._noise_std, (numbers.Number, np.ndarray)):
                return self._noise_std

            if isinstance(self._noise_std, str):
                filename = str(self._noise_std)
                if filename[-4:] == '.txt':
                    with open(filename, 'r') as f:
                        return float(f.read())
                return load_nifti(filename).get_data()

            self._logger.warning('Failed to obtain a noise std for this subject. We will continue with an std of 1.')
            return 1

        self._noise_std = _compute_noise_std()

        if is_scalar(self._noise_std):
            return self._noise_std
        else:
            return create_roi(self._noise_std, self.mask)

    def _preload_extra_protocol_items(self, extra_protocol):
        """Load all the extra protocol items that were defined by a filename."""
        if extra_protocol is None:
            return {}

        return_items = {}
        for key, value in extra_protocol.items():
            if isinstance(value, str):
                if value.endswith('.nii') or value.endswith('.nii.gz'):
                    loaded_val = load_nifti(value).get_data()
                else:
                    loaded_val = np.genfromtxt(value)
            else:
                loaded_val = value
            return_items[key] = loaded_val
        return return_items


class MockMRIInputData(SimpleMRIInputData):

    def __init__(self, protocol=None, signal4d=None, mask=None, nifti_header=None,
                 **kwargs):
        """A mock DMRI input data object that returns None for everything unless given.
        """
        super().__init__(protocol, signal4d, mask, nifti_header, **kwargs)

    def _get_constructor_args(self):
        """Get the constructor arguments needed to create a copy of this batch util using a copy constructor.

        Returns:
            tuple: args and kwargs tuple
        """
        args = [self._protocol, self.signal4d, self._mask, self.nifti_header]
        kwargs = {}
        return args, kwargs

    @property
    def nmr_voxels(self):
        return 0

    @property
    def observations(self):
        return self._observation_list

    @property
    def noise_std(self):
        return 1


class ROIMRIInputData(MRIInputData):

    def __init__(self, protocol, observations, mask, nifti_header, extra_protocol=None, gradient_deviations=None,
                 noise_std=None, volume_weights=None):
        """An MRI input data object initialized with ROI voxels, instead of full 3d/4d volumes.

        This can be helpful if you have a list of voxels (i.e. a ROI) you want to process. Using this class saves
        memory and time from constructing 4d volumes.

        Args:
            protocol (Protocol): The protocol object used as input data to the model
            observations (ndarray): The MRI data as a 2d list of observations
            mask (ndarray): The mask used to create the observations list
            nifti_header (nifti header): The header of the nifti file to use for writing the results.
            extra_protocol (Dict[str, val]): additional protocol items. Here one may additionally specify values to be
                used for the protocol parameters. These additional values can be scalars, vectors and/or 2d arrays.
                This in contrast to the ``protocol`` which only contains scalars and vectors. Items specified here will
                overwrite items from the protocol in the case of duplicated names. This parameter can for example be
                used to specify gradient volumes, instead of a gradient in the protocol, for example by specifying::

                    extra_protocol = {'g': np.array(...)}

                Per element, the input can be a scalar, a vector, an array, or a filename. If a filename is given
                we will try to interpret it again as a scalar, vector or array.
            gradient_deviations (str or ndarray): a gradient deviations matrix.
                The matrix can be provided in multiple formats:

                - an (n, 9) matrix with per voxel 9 values that constitute the gradient non-linearities
                    according to the HCP guidelines. (see
                    ``www.humanconnectome.org/storage/app/media/documentation/data_release/Q1_Release_Appendix_II.pdf``)
                    If given in this format, we will automatically add the identity matrix to it, as specified by the
                    HCP guidelines.
                - an (n, 3, 3) matrix with per voxel the deformation matrix. This will be used as given (i.e. no
                    identity matrix will be added to it like in the HCP format).
                - an (n, m, 3, 3) matrix with per voxel and per volume a deformation matrix. This will be used as
                    given.

            noise_std (number or ndarray): either None for automatic detection,
                or a scalar, or an 3d matrix with one value per voxel.

            volume_weights (ndarray): if given, a float matrix of the same size as the observations with per voxel
                and volume a weight in [0, 1]. If set, these weights are used during model fitting to weigh the
                objective function values per observation.
        """
        self._logger = logging.getLogger(__name__)
        self._observation_list = observations
        self._nifti_header = nifti_header
        self._mask = mask
        self._protocol = protocol
        self._extra_protocol = self._preload_extra_protocol_items(extra_protocol)
        self._noise_std = noise_std

        self._gradient_deviations = gradient_deviations
        self._gradient_deviations_list = None

        self._volume_weights = volume_weights
        self._volume_weights_list = None

        self._signal4d = None

        if protocol.length != 0:
            self._nmr_observations = protocol.length
        else:
            self._nmr_observations = observations.shape[1]

        if protocol.length != 0 and observations is not None and \
                observations.shape[1] != 0 and protocol.length != observations.shape[1]:
            raise ValueError('Length of the protocol ({}) does not equal the number of volumes ({}).'.format(
                protocol.length, observations.shape[1]))

        if self._volume_weights is not None and self._volume_weights.shape != self._observation_list.shape:
            raise ValueError('The dimensions of the volume weights does not match the dimensions of the observations.')

    @classmethod
    def from_input_data(cls, input_data):
        """Construct a ROI input data object from another input data object."""
        return cls(input_data.protocol, input_data.observations, input_data.mask, input_data.nifti_header,
                   extra_protocol=input_data.extra_protocol, gradient_deviations=input_data.gradient_deviations,
                   noise_std=input_data.noise_std, volume_weights=input_data.volume_weights)

    def has_input_data(self, parameter_name):
        try:
            self.get_input_data(parameter_name)
            return True
        except ValueError:
            return False

    def get_input_data(self, parameter_name):
        # todo compress
        if parameter_name in self._extra_protocol:
            return np.array(self._extra_protocol[parameter_name], copy=False)
        if parameter_name in self._protocol:
            return self._protocol[parameter_name]
        raise ValueError('No input data could be find for the parameter "{}".'.format(parameter_name))

    def copy_with_updates(self, **updates):
        """Create a copy of this input data, while setting some of the arguments to new values.

        Args:
            updates (kwargs): with constructor names.
        """
        arg_names = ['protocol', 'observations', 'mask', 'nifti_header']
        args = [self._protocol, self._observation_list, self._mask, self.nifti_header]
        kwargs = dict(extra_protocol=self._extra_protocol,
                      gradient_deviations=self._gradient_deviations,
                      noise_std=self._noise_std)

        for ind, arg_name in enumerate(arg_names):
            args[ind] = updates.get(arg_name, args[ind])

        for kwarg_name in kwargs.keys():
            kwargs[kwarg_name] = updates.get(kwarg_name, kwargs[kwarg_name])

        return self.__class__(*args, **kwargs)

    def get_subset(self, volumes_to_keep=None, volumes_to_remove=None):
        if (volumes_to_keep is not None) and (volumes_to_remove is not None):
            raise ValueError('You can not specify both the list with volumes to keep and volumes to remove. Choose one.')
        if (volumes_to_keep is None) and (volumes_to_remove is None):
            raise ValueError('Please specify either the list with volumes to keep or the list with volumes to remove.')

        if volumes_to_keep is not None:
            if is_scalar(volumes_to_keep):
                volumes_to_keep = [volumes_to_keep]
        elif volumes_to_remove is not None:
            if is_scalar(volumes_to_remove):
                volumes_to_remove = [volumes_to_remove]

            volumes_to_keep = list(range(self.nmr_observations))
            volumes_to_keep = [ind for ind in volumes_to_keep if ind not in volumes_to_remove]

        new_protocol = self.protocol
        if self.protocol is not None:
            new_protocol = self.protocol.get_new_protocol_with_indices(volumes_to_keep)

        new_observations = self.observations
        if self.observations is not None:
            new_observations = self.observations[..., volumes_to_keep]

        new_volume_weights = self._volume_weights
        if self._volume_weights is not None:
            new_volume_weights = new_volume_weights[..., volumes_to_keep]

        new_gradient_deviations = self._gradient_deviations
        if self._gradient_deviations is not None:
            if self._gradient_deviations.ndim > 3 and self._gradient_deviations.shape[1] == self.protocol.length:
                new_gradient_deviations = self._gradient_deviations[:, volumes_to_keep, ...]

        new_extra_protocol = {}
        for key, value in self._extra_protocol.items():
            value = np.array(value, copy=False)
            if value.ndim > 1 and value.shape[1] == self.protocol.length:
                value = value[:, volumes_to_keep, ...]
            new_extra_protocol[key] = value

        return self.copy_with_updates(protocol=new_protocol, observations=new_observations,
                                      gradient_deviations=new_gradient_deviations,
                                      volume_weights=new_volume_weights, extra_protocol=new_extra_protocol)

    @property
    def nmr_voxels(self):
        return self.observations.shape[0]

    @property
    def nmr_observations(self):
        return self._nmr_observations

    @property
    def signal4d(self):
        # return self._signal4d
        return restore_volumes(self._observation_list, self.mask)

    @property
    def nifti_header(self):
        return self._nifti_header

    @property
    def gradient_deviations(self):
        return self._gradient_deviations

    @property
    def volume_weights(self):
        return self._volume_weights_list

    @property
    def protocol(self):
        return self._protocol

    @property
    def extra_protocol(self):
        return self._extra_protocol

    @property
    def observations(self):
        return self._observation_list

    @property
    def mask(self):
        return self._mask

    @property
    def noise_std(self):
        return self._noise_std

    def _preload_extra_protocol_items(self, extra_protocol):
        """Load all the extra protocol items that were defined by a filename."""
        if extra_protocol is None:
            return {}

        return_items = {}
        for key, value in extra_protocol.items():
            if isinstance(value, str):
                if value.endswith('.nii') or value.endswith('.nii.gz'):
                    loaded_val = load_nifti(value).get_data()
                else:
                    loaded_val = np.genfromtxt(value)
            else:
                loaded_val = value

            if len(loaded_val.shape) > 3:
                loaded_val = create_roi(loaded_val, self._mask)

            return_items[key] = loaded_val
        return return_items


def load_input_data(volume_info, protocol, mask, extra_protocol=None, gradient_deviations=None,
                    noise_std=None, volume_weights=None):
    """Load and create the input data object for diffusion MRI modeling.

    Args:
        volume_info (string or tuple): Either an (ndarray, img_header) tuple or the full path
            to the volume (4d signal data).
        protocol (:class:`~mdt.protocols.Protocol` or str): A protocol object with the right protocol for the
            given data, or a string object with a filename to the given file.
        mask (ndarray, str): A full path to a mask file or a 3d ndarray containing the mask
        extra_protocol (Dict[str, val]): additional protocol items. Here one may additionally specify values to be
            used for the protocol parameters. These additional values can be scalars, vectors and/or volumes. This in
            contrast to the ``protocol`` which only contains scalars and vectors. Items specified here will overwrite
            items from the protocol in the case of duplicated names. This parameter can for example be used to specify
            gradient volumes, instead of a gradient in the protocol, for example by specifying::

                extra_protocol = {'g': np.array(...)}

            Per element, the input can be a scalar, a vector, an array, or a filename. If a filename is given
            we will try to interpret it again as a scalar, vector or array.

        gradient_deviations (str or ndarray): a gradient deviations matrix. If a string is given we will interpret it
                as a Nifti file. The matrix can be provided in multiple formats:

            - an (x, y, z, 9) matrix with per voxel 9 values that constitute the gradient non-linearities
                according to the HCP guidelines. (see
                ``www.humanconnectome.org/storage/app/media/documentation/data_release/Q1_Release_Appendix_II.pdf``).
                If given in this format, we will automatically add the identity matrix to it, as specified by the
                HCP guidelines.
            - an (x, y, z, 3, 3) matrix with per voxel the deformation matrix. This will be used as given (i.e. no
                identity matrix will be added to it like in the HCP format).
            - an (x, y, z, m, 3, 3) matrix with per voxel and per volume a deformation matrix. This will be used as
                given.

        noise_std (number or ndarray): either None for automatic detection,
            or a scalar, or an 3d matrix with one value per voxel.

        volume_weights (ndarray): if given, a float matrix of the same size as the volume with per voxel and volume
            a weight in [0, 1]. If set, these weights are used during model fitting to weigh the objective function
            values per observation.

    Returns:
        SimpleMRIInputData: the input data object containing all the info needed for diffusion MRI model fitting
    """
    protocol = load_protocol(protocol)
    mask = load_brain_mask(mask)

    if isinstance(volume_info, str):
        info = load_nifti(volume_info)
        signal4d = info.get_data()
        img_header = info.header
    else:
        signal4d, img_header = volume_info

    if isinstance(gradient_deviations, str):
        gradient_deviations = load_nifti(gradient_deviations).get_data()

    if isinstance(volume_weights, str):
        volume_weights = load_nifti(volume_weights).get_data()

    return SimpleMRIInputData(protocol, signal4d, mask, img_header, extra_protocol=extra_protocol, noise_std=noise_std,
                              gradient_deviations=gradient_deviations, volume_weights=volume_weights)
