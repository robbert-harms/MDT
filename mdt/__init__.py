import glob
import os
from contextlib import contextmanager
import logging.config as logging_config
import numpy as np
import six
from six import string_types
import logging


__author__ = 'Robbert Harms'
__date__ = "2015-03-10"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


VERSION = '0.7.24'
VERSION_STATUS = ''

_items = VERSION.split('-')
VERSION_NUMBER_PARTS = tuple(int(i) for i in _items[0].split('.'))
if len(_items) > 1:
    VERSION_STATUS = _items[1]
__version__ = VERSION


"""
The start_gui initialization of MDT.

We inlined all the MDT and MOT imports in the functions to prevent circular imports.

Also, when working with MOT and the python multiprocessing library we run into errors if load the OpenCL stack before
we load the python multiprocessing library. In particular, if we import PyOpencl (via MOT) before we start
the multiprocessing we will get an Out Of Memory exception when trying to create an kernel.
"""


def batch_fit(data_folder, batch_profile=None, subjects_selection=None, recalculate=False,
              models_to_fit=None, cascade_subdir=False,
              cl_device_ind=None, dry_run=False, double_precision=False):
    """Run all the available and applicable models on the data in the given folder.

    See the class AutoRun for more details and options.

    Setting the cl_device_ind has the side effect that it changes the current run time cl_device settings in the MOT
    toolkit.

    Args:
        data_folder (str): The data folder to process
        batch_profile (BatchProfile or str): the batch profile to use, or the name of a batch profile to load.
            If not given it is auto detected.
        subjects_selection (BatchSubjectSelection): the subjects to use for processing.
            If None all subjects are processed.
        recalculate (boolean): If we want to recalculate the results if they are already present.
        models_to_fit (list of str): A list of models to fit to the data. This overrides the models in
                the batch config.
        cascade_subdir (boolean): if we want to create a subdirectory for every cascade model.
            Per default we output the maps of cascaded results in the same directory, this allows reusing cascaded
            results for other cascades (for example, if you cascade BallStick -> Noddi you can use the BallStick results
            also for BallStick -> Charmed). This flag disables that behaviour and instead outputs the results of
            a cascade model to a subdirectory for that cascade. This does not apply recursive.
        cl_device_ind (int or list of int): the index of the CL device to use.
            The index is from the list from the function get_cl_devices().
        dry_run (boolean): a dry run will do no computations, but will list all the subjects found in the
            given directory.
        double_precision (boolean): if we would like to do the calculations in double precision

    Returns:
        The list of subjects we will calculate / have calculated.
    """
    import mdt.utils
    from mdt.model_fitting import BatchFitting

    if not utils.check_user_components():
        raise RuntimeError('Your components folder is not up to date. Please run the script mdt-init-user-settings.')

    batch_fitting = BatchFitting(data_folder, batch_profile=batch_profile, subjects_selection=subjects_selection,
                                 recalculate=recalculate, models_to_fit=models_to_fit, cascade_subdir=cascade_subdir,
                                 cl_device_ind=cl_device_ind, double_precision=double_precision)

    if dry_run:
        return batch_fitting.get_subjects_info()

    return batch_fitting.run()


def fit_model(model, problem_data, output_folder, optimizer=None,
              recalculate=False, only_recalculate_last=False, model_protocol_options=None,
              use_model_protocol_options=True, cascade_subdir=False,
              cl_device_ind=None, double_precision=False, gradient_deviations=None, noise_std='auto'):
    """Run the optimizer on the given model.

    Args:
        model (str or AbstractModel): An implementation of an AbstractModel that contains the model we want to optimize
            or the name of an model we load with get_model()
        problem_data (DMRIProblemData): the problem data object containing all the info needed for diffusion
            MRI model fitting
        output_folder (string): The path to the folder where to place the output, we will make a subdir with the
            model name in it.
        optimizer (AbstractOptimizer): The optimization routine to use.
        recalculate (boolean): If we want to recalculate the results if they are already present.
        only_recalculate_last (boolean):
            This is only of importance when dealing with CascadeModels.
            If set to true we only recalculate the last element in the chain (if recalculate is set to True, that is).
            If set to false, we recalculate everything. This only holds for the first level of the cascade.
        model_protocol_options (dict): specific model protocol options to use during fitting.
                This is for example used during batch fitting to limit the protocol for certain models.
                For instance, in the Tensor model we generally only want to use the lower b-values, or for S0 only
                the unweighted. Please note that this is merged with the options defined in the config file.
        use_model_protocol_options (boolean): if we want to use the model protocol options or not.
        cascade_subdir (boolean): if we want to create a subdirectory for the given model if it is a cascade model.
            Per default we output the maps of cascaded results in the same directory, this allows reusing cascaded
            results for other cascades (for example, if you cascade BallStick -> Noddi you can use the BallStick results
            also for BallStick -> Charmed). This flag disables that behaviour and instead outputs the results of
            a cascade model to a subdirectory for that cascade. This does not apply recursive.
        cl_device_ind (int or list): the index of the CL device to use. The index is from the list from the function
            utils.get_cl_devices(). This can also be a list of device indices.
        double_precision (boolean): if we would like to do the calculations in double precision
        gradient_deviations (str or ndarray): set of gradient deviations to use. In HCP WUMINN format.
        noise_std (None, double, ndarray, or 'auto'): the noise level standard deviation.
                The value can be either:
                    None: set to 1
                    double: use a single value for all voxels
                    ndarray: use a value per voxel (this should not be a roi list, it should be an actual volume
                        of the same size as the dataset)
                    string: a filename we will try to parse as a noise std
                    'auto': try to estimate the noise std

    Returns:
        the output of the optimization. If a cascade is given, only the results of the last model in the cascade is
        returned
    """
    import mdt.utils
    from mdt.model_fitting import ModelFit
    import six

    if gradient_deviations:
        if isinstance(gradient_deviations, six.string_types):
            gradient_deviations = load_nifti(gradient_deviations).get_data()

    if not utils.check_user_components():
        raise RuntimeError('Your components folder is not up to date. Please run the script mdt-init-user-settings.')

    model_fit = ModelFit(model, problem_data, output_folder, optimizer=optimizer, recalculate=recalculate,
                         only_recalculate_last=only_recalculate_last, model_protocol_options=model_protocol_options,
                         use_model_protocol_options=use_model_protocol_options,
                         cascade_subdir=cascade_subdir,
                         cl_device_ind=cl_device_ind, double_precision=double_precision,
                         gradient_deviations=gradient_deviations,
                         noise_std=noise_std)

    return model_fit.run()


def sample_model(model, problem_data, output_folder, sampler=None, recalculate=False,
                 model_protocol_options=None, use_model_protocol_options=True,
                 cl_device_ind=None, double_precision=False,
                 gradient_deviations=None, noise_std='auto', initialize=True, initialize_using=None):
    """Sample a single model. This does not accept cascade models, only single models.

    Args:
        model: the model to sample
        problem_data (DMRIProblemData): the problem data object, load with, for example, mdt.load_problem_data().
        output_folder (string): The path to the folder where to place the output, we will make a subdir with the
            model name in it (for the optimization results) and then a subdir with the samples output.
        sampler (AbstractSampler): the sampler to use
        recalculate (boolean): If we want to recalculate the results if they are already present.
        model_protocol_options (dict): specific model protocol options to use during fitting.
                This is for example used during batch fitting to limit the protocol for certain models.
                For instance, in the Tensor model we generally only want to use the lower b-values, or for S0 only
                the unweighted. Please note that this is merged with the options defined in the config file.
        use_model_protocol_options (boolean): if we want to use the model protocol options or not.
        cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
            utils.get_cl_devices().
        double_precision (boolean): if we would like to do the calculations in double precision
        gradient_deviations (str or ndarray): set of gradient deviations to use. In HCP WUMINN format.
                noise_std (None, double, ndarray or 'auto'): the noise level standard deviation.
                The value can be either:
                    None: set to 1
                    double: use a single value for all voxels
                    ndarray: use a value per voxel
                    string: a filename we will try to parse as a noise std
                    'auto': tries to estimate the noise std from the data
        initialize (boolean): If we want to initialize the sampler with optimization output.
            This assumes that the optimization results are in the folder:
                <output_folder>/<model_name>/
        initialize_using (None, str, or dict): If None, and initialize is True we will initialize from the
            optimization maps from a model with the same name. If a string is given and initialize is True we will
            interpret the string as a folder with the maps to load. If a dict is given and initialize is True we will
            initialize from the dict directly.

    Returns:
        dict: the samples per parameter as a numpy memmap.
    """
    import mdt.utils
    from mdt.models.cascade import DMRICascadeModelInterface
    from mdt.model_sampling import ModelSampling

    if gradient_deviations:
        if isinstance(gradient_deviations, six.string_types):
            gradient_deviations = load_nifti(gradient_deviations).get_data()

    if not utils.check_user_components():
        raise RuntimeError('Your components folder is not up to date. Please run the script mdt-init-user-settings.')

    sampling = ModelSampling(model, problem_data, output_folder,
                             sampler=sampler, recalculate=recalculate, cl_device_ind=cl_device_ind,
                             double_precision=double_precision,
                             model_protocol_options=model_protocol_options,
                             use_model_protocol_options=use_model_protocol_options,
                             gradient_deviations=gradient_deviations, noise_std=noise_std, initialize=initialize,
                             initialize_using=initialize_using)

    return sampling.run()


def collect_batch_fit_output(data_folder, output_dir, batch_profile=None, subjects_selection=None,
                             mask_name=None, symlink=False):
    """Load from the given data folder all the output files and put them into the output directory.

    If there is more than one mask file available the user has to choose which mask to use using the mask_name
    keyword argument. If it is not given an error is raised.

    The results for the chosen mask it placed in the output folder per subject. Example:
        <output_dir>/<subject_id>/<results>

    Args:
        data_folder (str): The data folder with the output files
        output_dir (str): The path to the output folder where all the files will be put.
        batch_profile (BatchProfile class or str): the batch profile to use, can also be the name
            of a batch profile to load. If not given it is auto detected.
        subjects_selection (BatchSubjectSelection): the subjects to use for processing.
            If None all subjects are processed.
        mask_name (str): the mask to use to get the output from
        symlink (boolean): only available under Unix OS's. Creates a symlink instead of copying.
    """
    from mdt.batch_utils import collect_batch_fit_output
    collect_batch_fit_output(data_folder, output_dir, batch_profile=batch_profile,
                             subjects_selection=subjects_selection,
                             mask_name=mask_name, symlink=symlink)


def run_function_on_batch_fit_output(data_folder, func, batch_profile=None, subjects_selection=None):
    """Run a function on the output of a batch fitting routine.

    This enables you to run a function on every model output from every subject. The python function should accept
    as single argument an instance of the class BatchFitSubjectOutputInfo.

    Args:
        data_folder (str): The data folder with the output files
        func (python function): the python function we should call for every map and model.
            This should accept as single parameter a BatchFitSubjectOutputInfo.
        batch_profile (BatchProfile or str): the batch profile to use, can also be the name
            of a batch profile to load. If not given it is auto detected.
        subjects_selection (BatchSubjectSelection): the subjects to use for processing.
            If None all subjects are processed.

    Returns:
        dict: indexed by subject->model_name->mask_name, values are the return values of the user function
    """
    from mdt.batch_utils import run_function_on_batch_fit_output
    return run_function_on_batch_fit_output(data_folder, func, batch_profile=batch_profile,
                                            subjects_selection=subjects_selection)


def estimate_noise_std(problem_data, estimation_cls_name=None):
    """Estimate the noise standard deviation.

    Args:
        problem_data (DMRIProblemData): the problem data we can use to do the estimation
        estimation_cls_name (str): the name of the estimation class to load. If none given we try each defined in the
            current config.

    Returns:
        the noise std estimated from the data. This can either be a single float, or an ndarray.

    Raises:
        NoiseStdEstimationNotPossible: if the noise could not be estimated
    """
    from mdt.utils import estimate_noise_std
    return estimate_noise_std(problem_data, estimation_cls_name=estimation_cls_name)


def get_cl_devices():
    """Get a list of all CL devices in the system.

    The indices of the devices can be used in the model fitting functions for 'cl_device_ind'.

    Returns:
        A list of CLEnvironments, one for each device in the system.
    """
    from mdt.utils import get_cl_devices
    return get_cl_devices()


def get_device_ind(device_type='FIRST_GPU'):
    """Convenience function to get the device type for a particular type of device.

    If device type is not one of the defined types, we default to 'FIRST_GPU'.

    Args:
        device_type (str): choose one of 'FIRST_GPU', 'ALL_GPU', 'FIRST_CPU', 'ALL_CPU', 'ALL'

    Returns
        list: the list of device indices for the requested type of devices.
    """
    supported_types = ['FIRST_GPU', 'ALL_GPU', 'FIRST_CPU', 'ALL_CPU', 'ALL']
    if device_type is None or device_type not in supported_types:
        device_type = 'FIRST_GPU'

    from mot.cl_environments import CLEnvironmentFactory
    devices = CLEnvironmentFactory.all_devices()

    if device_type == 'ALL':
        return range(0, len(devices))

    indices = []
    if 'CPU' in device_type:
        for ind, dev in enumerate(devices):
            if dev.is_cpu:
                indices.append(ind)
    elif 'GPU' in device_type:
        for ind, dev in enumerate(devices):
            if dev.is_gpu:
                indices.append(ind)

    if not indices:
        raise ValueError('No suitable CPU or GPU index found.')

    if 'first' in device_type:
        return indices[0]
    return indices


def load_problem_data(volume_info, protocol, mask, static_maps=None, dtype=np.float32):
    """Load and create the problem data object that can be given to a model

    Args:
        volume_info (string): Either an (ndarray, img_header) tuple or the full path to the volume (4d signal data).
        protocol (Protocol or string): A protocol object with the right protocol for the given data,
            or a string object with a filename to the given file.
        mask (string): A full path to a mask file that can optionally be used. If None given, no mask is used.
        static_maps (Dict[str, val]): the dictionary with per static map the value to use.
            The value can either be an 3d or 4d ndarray, a single number or a string. We will convert all to the
            right format.
        dtype (dtype) the datatype in which to load the signal volume.

    Returns:
        The Problem data, in the ProblemData container object.
    """
    from mdt.utils import load_problem_data
    return load_problem_data(volume_info, protocol, mask, static_maps=static_maps, dtype=dtype)


def load_protocol_bval_bvec(bvec=None, bval=None, bval_scale='auto'):
    """Create a protocol out of a bvac filename and a bvec filename.

    Args:
        bval (string): The bval filename
        bvec (string): The bvec filename
        bval_scale (double): The scale by which to scale the values in the bval file.

    Returns:
        A Protocol object.
    """
    from mdt.protocols import load_bvec_bval
    return load_bvec_bval(bvec, bval, bval_scale=bval_scale)


def load_protocol(filename):
    """Load an protocol from the given protocol file, with as column names the given list of names.

    Args:
        filename (string):
            The filename of the protocol file to load. This should be a comma seperated, or tab delimited file
            with equal length columns. The column names can go on the tab and should be comma or space seperated.

    Returns:
        An protocol with all the columns loaded.
    """
    from mdt.protocols import load_protocol
    return load_protocol(filename)


def auto_load_protocol(directory, protocol_options=None, bvec_fname=None, bval_fname=None, bval_scale='auto'):
    """Load a protocol from the given directory.

    This will first try to load the first .prtcl file found. If none present it will try to find bval and bvec files
    to load. If present it will also try to load additional columns.

    For more detail see auto_load_protocol in the protocols module.

    Args:
        directory (str): the directory to load the protocol from
        protocol_options (dict): mapping protocol items to filenames (as a subpath of the given directory)
            or mapping them to values (one value or one value per bvec line)
        bvec_fname (str): if given, the filename of the bvec file (as a subpath of the given directory)
        bval_fname (str): if given, the filename of the bvec file (as a subpath of the given directory)

    Returns:
        Protocol: a loaded protocol file.

    Raises:
        ValueError: if not enough information could be found. (No protocol or no bvec/bval combo).
    """
    import mdt.protocols
    return mdt.protocols.auto_load_protocol(directory, protocol_options=protocol_options,
                                            bvec_fname=bvec_fname, bval_fname=bval_fname,
                                            bval_scale=bval_scale)


def write_protocol(protocol, fname, columns_list=None):
    """Write a protocol to a file.

    Args:
        protocol (Protocol): The protocol object information to write
        fname (string): The filename to write to
        columns_list (list, optional): Only write these columns (and in this order).
    """
    from mdt.protocols import write_protocol
    write_protocol(protocol, fname, columns_list=columns_list)


def write_protocol_bvec_bval(protocol, bvec_fname, bval_fname, column_based=True, bval_scale='auto'):
    """Write a protocol to two files bval and bvec

    Args:
        protocol (Protocol): The protocol object information to write
        bvec_fname (string): The filename of the b-vector
        bval_fname (string): The filename of the b values
        column_based (boolean, optional, default true): If true, this supposes that the bvec (the vector file)
            will have 3 rows (gx, gy, gz)
            and will be space or tab seperated and that the bval file (the b values) are one one single line
            with space or tab separated b values.
        bval_scale (double or 'auto'): the amount by which we want to scale (multiply) the b-values.
            The default is auto, this checks if the first b-value is higher than 1e4 and if so multiplies it by
            1e-6 else multiplies by 1.
    """
    from mdt.protocols import write_bvec_bval
    write_bvec_bval(protocol, bvec_fname, bval_fname, column_based=column_based, bval_scale=bval_scale)


def create_median_otsu_brain_mask(dwi_info, protocol, output_fname=None, **kwargs):
    """Create a brain mask and optionally write it.

    It will always return the mask. If output_fname is set it will also write the mask.

    Args:
        dwi_info (string or (image, header) pair or image):
            - the filename of the input file;
            - or a tuple with as first index a ndarray with the DWI and as second index the header;
            - or only the image as an ndarray
        protocol (string or Protocol): The filename of the protocol file or a Protocol object
        output_fname (string): the filename of the output file. If None, no output is written.
            If dwi_info is only an image also no file is written.
        **kwargs: the additional arguments for median_otsu.

    Returns:
        ndarray: The created brain mask
    """
    from mdt.masking import create_median_otsu_brain_mask, create_write_median_otsu_brain_mask

    if output_fname:
        if not isinstance(dwi_info, (string_types, tuple, list)):
            raise ValueError('No header obtainable, can not write the brain mask.')
        return create_write_median_otsu_brain_mask(dwi_info, protocol, output_fname, **kwargs)
    return create_median_otsu_brain_mask(dwi_info, protocol, **kwargs)


def create_blank_mask(volume4d_path, output_fname):
    """Create a blank mask for the given 4d volume.

    Sometimes you want to use all the voxels in the given dataset, without masking any voxel. Since the optimization
    routines require a mask, you have to submit one. The solution is to use a blank mask, that is, a mask that
    masks nothing.

    Args:
        volume4d_path (str): the path to the 4d volume you want to create a blank mask for
        output_fname (str): the path to the result mask
    """
    volume_info = load_nifti(volume4d_path)
    mask = np.ones(volume_info.shape[:3])
    write_image(output_fname, mask, volume_info.get_header())


def load_brain_mask(brain_mask_fname):
    """Load the brain mask from the given file.

    Args:
        brain_mask_fname (string): The filename of the brain mask to load.

    Returns:
        The loaded brain mask data
    """
    from mdt.utils import load_brain_mask
    return load_brain_mask(brain_mask_fname)


def concatenate_mri_sets(items, output_volume_fname, output_protocol_fname, overwrite_if_exists=False):
    """Concatenate two or more DMRI datasets. Normally used to concatenate different DWI shells into one image.

    This writes a single volume and a single protocol file.

    Args:
        items (tuple): A tuple of dicts with volume filenames and protocol filenames
            (
             {'volume': volume_fname,
              'protocol': protocol_filename
             }, ...
            )
        output_volume_fname (string): The name of the output volume
        output_protocol_fname (string): The name of the output protocol file
        overwrite_if_exists (boolean, optional, default false): Overwrite the output files if they already exists.
    """
    import os
    import nibabel as nib
    from mdt.utils import concatenate_two_mri_measurements
    from mdt.protocols import write_protocol

    if not items:
        return

    if os.path.exists(output_volume_fname) and os.path.exists(output_protocol_fname) and not overwrite_if_exists:
        return

    to_concat = []
    nii_header = None

    for e in items:
        signal_img = nib.load(e['volume'])
        signal4d = signal_img.get_data()
        nii_header = signal_img.get_header()

        protocol = load_protocol(e['protocol'])
        to_concat.append((protocol, signal4d))

    protocol, signal4d = concatenate_two_mri_measurements(to_concat)
    nib.Nifti1Image(signal4d, None, nii_header).to_filename(output_volume_fname)
    write_protocol(protocol, output_protocol_fname)


def volume_merge(volume_paths, output_fname, sort=False):
    """Merge a list of volumes on the 4th dimension. Writes the result as a file.

    You can enable sorting the list of volume names based on a natural key sort. This is
    the most convenient option in the case of globbing files. By default this behaviour is disabled.

    Example usage with globbing:
        mdt.volume_merge(glob.glob('*.nii'), 'merged.nii.gz', True)

    Args:
        volume_paths (list of str): the list with the input filenames
        output_fname (str): the output filename
        sort (boolean): if true we natural sort the list of DWI images before we merge them. If false we don't.
            The default is True.

    Returns:
        list of str: the list with the filenames in the order of concatenation.
    """
    import re
    import nibabel as nib

    images = []
    header = None

    if sort:
        def natural_key(_str):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', _str)]
        volume_paths.sort(key=natural_key)

    for volume in volume_paths:
        nib_container = nib.load(volume)
        header = header or nib_container.get_header()
        image_data = nib_container.get_data()

        if len(image_data.shape) < 4:
            image_data = np.expand_dims(image_data, axis=3)

        images.append(image_data)

    combined_image = np.concatenate(images, axis=3)
    nib.Nifti1Image(combined_image, None, header).to_filename(output_fname)

    return volume_paths


def view_results_slice(data,
                       dimension=None,
                       slice_ind=None,
                       maps_to_show='auto',
                       map_titles=None,
                       general_plot_options=None,
                       map_plot_options=None,
                       font_size=None,
                       to_file=None,
                       block=True,
                       maximize=False,
                       window_title=None,
                       axis_options=None,
                       nmr_colorbar_axis_ticks=None,
                       show_sliders=None,
                       figure_options=None,
                       grid_layout=None,
                       article_modus=False,
                       rotate_images=None):
    """View from the given results the given slice.

    See MapsVisualizer.show() for most of the the options. The special options are listed in the section Args

    Args:
        article_modus (boolean): If set to true we set most of the options as such that the data is rendered better
            for in use of a paper. Sets
                axis_options='off'
                font_size=36
                nmr_colorbar_axis_ticks=4
                show_sliders=False
            You can overwrite these again by specifying one of these options directly.

    Returns:
        MapViewSettings: the settings set by the user in the viewer.
    """
    from mdt.visualization import MapsVisualizer

    general_plot_options = general_plot_options or {}

    if 'cmap' not in general_plot_options:
        general_plot_options.update({'cmap': 'hot'})

    if isinstance(data, string_types):
        map_names = None
        if maps_to_show:
            if maps_to_show == 'auto':
                map_names = results_preselection_names(data)
            else:
                map_names = maps_to_show
        results_total = load_volume_maps(data, map_names=map_names)
    else:
        results_total = data

    if maps_to_show:
        if maps_to_show == 'auto':
            maps_to_show = results_preselection_names(data)
        else:
            results_total = {k: results_total[k] for k in maps_to_show}

    if article_modus:
        axis_options = axis_options or 'off'
        font_size = font_size or 36
        nmr_colorbar_axis_ticks = nmr_colorbar_axis_ticks or 4
        show_sliders = show_sliders or False

    viz = MapsVisualizer(results_total)
    if font_size:
        viz.font_size = font_size
    return viz.show(
        dimension=dimension,
        slice_ind=slice_ind,
        maps_to_show=maps_to_show,
        map_titles=map_titles,
        general_plot_options=general_plot_options,
        map_plot_options=map_plot_options,
        to_file=to_file,
        block=block,
        maximize=maximize,
        window_title=window_title,
        axis_options=axis_options,
        nmr_colorbar_axis_ticks=nmr_colorbar_axis_ticks,
        show_sliders=show_sliders,
        figure_options=figure_options,
        grid_layout=grid_layout,
        rotate_images=rotate_images)


def results_preselection_names(data):
    """Generate a list of useful map names to display.

    This is primarily to be used as argument to the parameter 'maps_to_show' of the function view_results_slice.

    Args:
        data (str or dict or list of str): either a directory or a dictionary of results or a list of map names.

    Returns:
        list of str: the list of useful/filtered map names.
    """
    keys = []
    if isinstance(data, string_types):
        for extension in ('.nii', '.nii.gz'):
            for f in glob.glob(os.path.join(data, '*' + extension)):
                keys.append(os.path.basename(f)[0:-len(extension)])
    elif isinstance(data, dict):
        keys = data.keys()
    else:
        keys = data

    filter_match = ('.vec', '.d', '.sigma', 'AIC', 'Errors.mse', 'Errors.sse', '.eigen_ranking', 'SignalEstimates')
    return list(sorted(filter(lambda v: all(m not in v for m in filter_match), keys)))


def block_plots():
    """A small function to block the plots made by matplotlib.

    This basically only calls plt.show()
    """
    import matplotlib.pyplot as plt
    plt.show()


def view_result_samples(data, **kwargs):
    """View the samples from the given results set.

    Args:
        data (string or dict): The location of the maps to load the samples from, or the samples themselves.
        kwargs (dict): see SampleVisualizer for all the supported keywords
    """
    from mdt.visualization import SampleVisualizer
    from mdt.utils import load_samples

    if isinstance(data, string_types):
        data = load_samples(data)

    if not kwargs.get('voxel_ind'):
        kwargs.update({'voxel_ind': data[list(data.keys())[0]].shape[0] / 2})
    SampleVisualizer(data).show(**kwargs)


def load_volume(volume_fname, ensure_4d=True):
    """Load the diffusion weighted image data from the given volume filename.

    This does not perform any data type changes, so the input may not be in float64. If you call this function
    to satisfy load_problem_data() this is not a problem.

    Args:
        volume_fname (string): The filename of the volume to load.
        ensure_4d (boolean): if True we ensure that the data matrix is in 4d.

    Returns:
        a tuple with (data, header) for the given file.
    """
    from mdt.utils import load_volume
    return load_volume(volume_fname, ensure_4d=ensure_4d)


def load_nifti(nifti_volume):
    """Load and return a nifti file.

    A more general function than load_dwi which is meant for raw diffusion images.

    Args:
        nifti_volume (string): The filename of the volume to load.

    Returns:
        nib image
    """
    import nibabel as nib
    return nib.load(nifti_volume)


def make_path_joiner(*folder):
    """Generates and returns an instance of utils.PathJoiner to quickly join pathnames.

    Returns:
        An instance of utils.PathJoiner for easy path manipulation.
    """
    from mdt.utils import PathJoiner
    return PathJoiner(*folder)


def create_roi(data, brain_mask):
    """Create and return the region of interest of the given brain volume and mask

    Args:
        data (string or ndarray): a brain volume with four dimensions (x, y, z, w)
            where w is the length of the protocol, or a list, tuple or dictionary with volumes or a string
            with a filename of a dataset to load.
        brain_mask (ndarray or str): the mask indicating the region of interest, dimensions: (x, y, z) or the string
            to the brain mask to load

    Returns:
        Signal lists for each of the given volumes. The axis are: (voxels, protocol)
    """
    from mdt.utils import create_roi
    return create_roi(data, brain_mask)


def create_slice_roi(brain_mask_fname, roi_dimension, roi_slice, output_fname, overwrite_if_exists=False):
    """Create a region of interest out of the given brain mask by taking one specific slice out of the mask.

    This will both write and return the created slice ROI.

    We need a filename as input brain mask since we need the header of the file to be able to write the output file
    with the same header.

    Args:
        brain_mask (string): The filename of the brain_mask used to create the new brain mask
        roi_dimension (int): The dimension to take a slice out of
        roi_slice (int): The index on the given dimension.
        output_fname (string): The output filename
        overwrite_if_exists (boolean, optional, default false): If we want to overwrite the file if it already exists

    Returns:
        A brain mask of the same dimensions as the original mask, but with only one slice set to one.
    """
    import os
    import nibabel as nib
    from mdt.utils import create_slice_roi

    if os.path.exists(output_fname) and not overwrite_if_exists:
        return load_brain_mask(output_fname)

    if not os.path.isdir(os.path.dirname(output_fname)):
        os.makedirs(os.path.dirname(output_fname))

    brain_mask_img = nib.load(brain_mask_fname)
    brain_mask = brain_mask_img.get_data()
    img_header = brain_mask_img.get_header()
    roi_mask = create_slice_roi(brain_mask, roi_dimension, roi_slice)
    write_image(output_fname, roi_mask, img_header)
    return roi_mask


def write_image(fname, data, header):
    """Write the given data with the given header to the given file.

    Args:
        fname (str): The output filename
        data (ndarray): The data to write
        header (nibabel header): The header to use
    """
    import nibabel as nib
    nib.Nifti1Image(data, None, header).to_filename(fname)


def restore_volumes(data, brain_mask, with_volume_dim=True):
    """Restore the given data to a whole brain volume

    The data can be a list, tuple or dictionary or directly a two dimensional list of data points

    This is the inverse function of create_roi().

    Args:
        data: the data as a x dimensional list of voxels, or, a list, tuple, or dict of those voxel lists
        brain_mask: the brain_mask which was used to generate the data list
        with_volume_dim (boolean): If true we return values with 4 dimensions. The extra dimension is for
            the volume index. If false we return 3 dimensions.

    Returns:
        Either a single whole volume, a list, tuple or dict of whole volumes, depending on the given data.
        If with_volume_ind_dim is set we return values with 4 dimensions. (x, y, z, 1). If not set we return only
        three dimensions.
    """
    from mdt.utils import restore_volumes
    return restore_volumes(data, brain_mask, with_volume_dim=with_volume_dim)


def write_trackmark_rawmaps(data, output_folder, maps_to_convert=None):
    """Convert the given nifti files in the input folder to rawmaps in the output folder.

    Args:
        data (str or dict): the name of the input folder, of a dictionary with maps to save.
        output_folder (str): the name of the output folder. Defaults to <input_folder>/trackmark.
        maps_to_convert (list): the list with the names of the maps we want to convert (without the extension).
    """
    from mdt.IO import TrackMark

    if isinstance(data, six.string_types):
        volumes = load_volume_maps(data, map_names=maps_to_convert)
    else:
        volumes = data
        if maps_to_convert:
            volumes = {k: v for k, v in volumes.items() if k in maps_to_convert}
    TrackMark.write_rawmaps(output_folder, volumes)


def write_trackmark_tvl(output_tvl, vector_directions, vector_magnitudes, tvl_header=(1, 1.8, 0, 0)):
    """Write a list of vector directions with corresponding magnitude to a trackmark TVL file.

    Note that the length of the vector_directions and vector_magnitudes should correspond to each other. Next, we only
    use the first three elements in both lists.

    Args:
        output_tvl (str): the name of the output tvl
        vector_directions (list of str/ndarray): a list of 4d volumes with per voxel the normalized vector direction
        vector_magnitudes (list of str/ndarray): a list of 4d volumes with per voxel the vector magnitude.
        tvl_header (list or tuple): The list with header arguments for writing the TVL. See IO.TrackMark for specifics.
    """
    from mdt.IO import TrackMark
    if len(vector_directions) != len(vector_magnitudes):
        raise ValueError('The length of the list of vector directions does not '
                         'match with the length of the list of vector magnitudes.')
    TrackMark.write_tvl_direction_pairs(output_tvl, tvl_header, list(zip(vector_directions, vector_magnitudes))[:3])


def sort_maps(maps_to_sort_on, extra_maps_to_sort=None, reversed_sort=False, sort_index_map=None):
    """Sort the given maps on the maps to sort on.

    This first creates a sort matrix to index the maps in sorted order per voxel. Next, it creates the output
    maps for the maps we sort on. If extra_maps_to_sort is given it should be of the same length as the maps_to_sort_on.

    Args:
        maps_to_sort_on (list): a list of string (filenames) or ndarrays we will load and compare
        extra_maps_to_sort (list) an additional list we will sort based on the indices in maps_to_sort. This should
            be of the same length as maps_to_sort_on.
        reversed_sort (boolean): if we want to sort from large to small instead of small to large.
        sort_index_map (ndarray): if given we use this sort index map instead of generating one by sorting the
            maps_to_sort_on.

    Returns:
        tuple: the first element is the list of sorted volumes, the second the list of extra sorted maps and the
            last is the sort index map used.
    """
    def load_maps(map_list):
        tmp = []
        for data in map_list:
            if isinstance(data, string_types):
                tmp.append(load_volume(data)[0])
            else:
                tmp.append(data)
        return tmp

    maps_to_sort_on = load_maps(maps_to_sort_on)
    if extra_maps_to_sort:
        extra_maps_to_sort = load_maps(extra_maps_to_sort)

        if len(extra_maps_to_sort) != len(maps_to_sort_on):
            raise ValueError('The length of the maps to sort on and the extra maps to sort do not match.')

    from mdt.utils import create_sort_matrix, sort_volumes_per_voxel

    if sort_index_map is None:
        sort_index_map = create_sort_matrix(np.concatenate([m for m in maps_to_sort_on], axis=3),
                                            reversed_sort=reversed_sort)
    elif isinstance(sort_index_map, string_types):
        sort_index_map = np.round(load_volume(sort_index_map)[0]).astype(np.int64)

    sorted_maps = sort_volumes_per_voxel(maps_to_sort_on, sort_index_map)
    if extra_maps_to_sort:
        sorted_extra_maps = sort_volumes_per_voxel(extra_maps_to_sort, sort_index_map)
        return sorted_maps, sorted_extra_maps, sort_index_map

    return sorted_maps, [], sort_index_map


def load_volume_maps(directory, map_names=None):
    """Read a number of Nifti volume maps that were written using write_volume_maps.

    Args:
        directory: the directory from which we want to read a number of maps
        map_names: the names of the maps we want to load. If given we only load and return these maps.

    Returns:
        dict: A dictionary with the volumes. The keys of the dictionary are the filenames (without the extension) of the
            files in the given directory.
    """
    from mdt.IO import Nifti
    return Nifti.read_volume_maps(directory, map_names=map_names)


def get_volume_names(directory):
    """Get the names of the Nifti volume maps in the given directory.

    Args:
        directory: the directory to get the names of the available maps from.

    Returns:
        list: A list with the names of the volumes.
    """
    from mdt.IO import Nifti
    return list(sorted(Nifti.volume_names_generator(directory)))


def write_volume_maps(maps, directory, header, overwrite_volumes=True):
    """Write a dictionary with maps to the given directory using the given header.

    Args:
        maps (dict): The maps with as keys the map names and as values 3d or 4d maps
        directory (str): The dir to write to
        header: The Nibabel Image Header
        overwrite_volumes (boolean): If we want to overwrite the volumes if they are present.
    """
    from mdt.IO import Nifti
    return Nifti.write_volume_maps(maps, directory, header, overwrite_volumes=overwrite_volumes)


def get_list_of_single_models():
    """Get a list of all available single models

    Returns:
        list of str: A list of all available single model names.
    """
    from mdt.components_loader import SingleModelsLoader
    return SingleModelsLoader().list_all()


def get_list_of_cascade_models():
    """Get a list of all available cascade models

    Returns:
        list of str: A list of available cascade models
    """
    from mdt.components_loader import CascadeModelsLoader
    return CascadeModelsLoader().list_all()


def get_models_list():
    """Get a list of all available models, single and cascade.

    Returns:
        list of str: A list of available model names.
    """
    l = get_list_of_cascade_models()
    l.extend(get_list_of_single_models())
    return l


def get_models_meta_info():
    """Get the meta information tags for all the models returned by get_models_list()

    Returns:
        dict of dict: The first dictionary indexes the model names to the meta tags, the second holds the meta
            information.
    """
    from mdt.components_loader import SingleModelsLoader, CascadeModelsLoader
    sml = SingleModelsLoader()
    cml = CascadeModelsLoader()

    meta_info = {}
    for model_loader in (sml, cml):
        models = model_loader.list_all()
        for model in models:
            meta_info.update({model: model_loader.get_meta_info(model)})
    return meta_info


def get_model(model_name, **kwargs):
    """Load one of the available models.

    Args:
        model_name (str): One of the models from get_list_of_single_models() or get_list_of_cascade_models()
        **kwargs: Extra keyword arguments used for the initialization of the model

    Returns:
        Either a cascade model or a single model. In any case, a model that can be given to the fit_model function.
    """
    from mdt.components_loader import get_model
    return get_model(model_name, **kwargs)


def get_batch_profile(batch_profile_name, *args, **kwargs):
    """Load one of the batch profiles.

    This is short for load_component('batch_profiles', batch_profile_name).

    Args:
        batch_profile_name (str): the name of the batch profile to load

    Returns:
        BatchProfile: the batch profile for use in batch fitting routines.
    """
    return load_component('batch_profiles', batch_profile_name, *args, **kwargs)


def load_component(component_type, component_name, *args, **kwargs):
    """Load the class indicated by the given component type and name.

    Args:
        component_type (str): the type of component, for example 'batch_profiles' or 'parameters'
        component_name (str): the name of the component to load
        *args: passed to the component
        **kwargs: passed to the component

    Returns:
        the loaded component
    """
    from mdt.components_loader import load_component
    return load_component(component_type, component_name, *args, **kwargs)


def split_dataset(input_fname, split_dimension, split_index, output_folder=None):
    """Split the given dataset along the given dimension on the given index.

    Args:
        dataset (ndarray, list, tuple or dict): The single or list of volume which to split in two
        split_dimension (int): The dimension along which to split the dataset
        split_index (int): The index on the given dimension to split the volume(s)

    Returns:
        If dataset is a single volume return the two volumes that when concatenated give the original volume back.
        If it is a list, tuple or dict return two of those with exactly the same indices but with each holding one half
        of the splitted data.
    """
    import os
    import nibabel as nib
    import mdt.utils
    from mdt.IO import Nifti

    if output_folder is None:
        output_folder = os.path.dirname(input_fname)
    dataset = nib.load(input_fname)
    data = dataset.get_data()
    header = dataset.get_header()
    split = utils.split_dataset(data, split_dimension, split_index)

    basename = os.path.basename(input_fname).split('.')[0]
    length = data.shape[split_dimension]
    lengths = (repr(0) + 'to' + repr(split_index-1), repr(split_index) + 'to' + repr(length-1))

    volumes = {}
    for ind, v in enumerate(split):
        volumes.update({str(basename) + '_split_' + str(split_dimension) + '_' + lengths[ind]: v})

    Nifti.write_volume_maps(volumes, output_folder, header)


def extract_volumes(input_volume_fname, input_protocol, output_volume_fname, output_protocol, protocol_indices):
    """Extract volumes from the given volume and save them to separate files.

    This will index the given input volume in the 4th dimension, as is usual in multi shell DWI files.

    Args:
        input_volume_fname (str): the input volume from which to get the specific volumes
        input_protocol (str or Protocol): the input protocol, either a file or preloaded protocol object
        output_volume_fname (str): the output filename for the selected volumes
        output_protocol (str): the output protocol for the selected volumes
        protocol_indices (list): the desired indices, indexing the input_volume
    """
    from mdt.data_loaders.protocol import autodetect_protocol_loader
    import mdt.protocols

    input_protocol = autodetect_protocol_loader(input_protocol).get_protocol()

    new_protocol = input_protocol.get_new_protocol_with_indices(protocol_indices)
    protocols.write_protocol(new_protocol, output_protocol)

    input_volume = load_volume(input_volume_fname)
    image_data = input_volume[0][..., protocol_indices]
    write_image(output_volume_fname, image_data, input_volume[1])


def apply_mask(volume, mask, inplace=True):
    """Apply a mask to the given input.

    Args:
        volume (str, ndarray, list, tuple or dict): The input file path or the image itself or a list, tuple or
            dict.
        mask (str or ndarray): The filename of the mask or the mask itself
        inplace (boolean): if True we apply the mask in place on the volume image. If false we do not.

    Returns:
        Depending on the input either a singla image of the same size as the input image, or a list, tuple or dict.
        This will set for all the output images the the values to zero where the mask is zero.
    """
    import mdt.utils
    return mdt.utils.apply_mask(volume, mask, inplace=inplace)


def apply_mask_to_file(input_fname, mask, output_fname=None):
    """Apply a mask to the given input (nifti) file.

    If no output filename is given, the input file is overwritten.

    Args:
        input_fname (str): The input file path
        mask (str or ndarray): The mask to use
        output_fname (str): The filename for the output file (the masked input file).
    """
    from mdt.data_loaders.brain_mask import autodetect_brain_mask_loader
    mask = autodetect_brain_mask_loader(mask).get_data()

    if output_fname is None:
        output_fname = input_fname

    image_info = load_volume(input_fname)
    mask = mask.reshape(mask.shape + (image_info[0].ndim - mask.ndim) * (1,))
    masked = image_info[0]
    masked *= mask
    write_image(output_fname, masked, image_info[1])


def init_user_settings(pass_if_exists=True, keep_config=True):
    """Initializes the user settings folder using a skeleton.

    This will create all the necessary directories for adding components to MDT. It will also create a basic
    configuration file for setting global wide MDT options.

    Each MDT version will have it's own sub-directory in the choosen folder.

    Args:
        pass_if_exists (boolean): if the folder for this version already exists, do we want to pass_if_exists yes or no.
        keep_config (boolean): if the folder for this version already exists, do we want to keep the config
            file yes or no. This only holds for the config file.

    Returns:
        the path the user settings skeleton was written to
    """
    from mdt.utils import init_user_settings
    return init_user_settings(pass_if_exists, keep_config)


@contextmanager
def config_context(config, config_clear=()):
    """Creates a temporary configuration context with the given config.

    This will temporarily alter the given configuration keys to the given values. After the context is executed
    the configuration will revert to the original settings.

    Example usage:
        config = '''
        optimization_settings:
            general:
                optimizers:
                    -   name: 'NMSimplex'
                        patience: 10
        '''
        with mdt.config_context(mdt.yaml_string_to_dict(config)):
            mdt.fit_model(...)

        This loads the configuration from a YAML string, converts it to a dict using the function
        mdt.yaml_string_to_dict() and then uses that config dict as context for the optimization.

    The config_clear list can be used to clear configuration items before those items are updated with
    the new values. This is necessary since the new config dictionary can only add or update items and can not
    remove them. This will effectively traverse the paths in the list of given paths and pop the given item from
    the dictionary.

    Args:
        config (dict): the configuration as a dictionary
        config_clear (list of list): the configuration items to clear before we update with the config options.
            Example: [['layer_0', 'layer_1'], ['some_other', 'somewhere']] will clear two paths (remove them from the
            config dict before adding the new values).
    """
    import copy
    import mdt.configuration
    from mdt.utils import recursive_merge_dict
    old_config = copy.deepcopy(mdt.configuration.config)
    new_config = mdt.configuration.config

    def remove_from_dict(d, path):
        for p in path[:-1]:
            d = d[p]
        d.pop(path[-1])

    for config_path in config_clear:
        remove_from_dict(new_config, config_path)

    mdt.configuration.config = recursive_merge_dict(new_config, config, in_place=True)
    yield
    mdt.configuration.config = old_config


def yaml_string_to_dict(yaml_str):
    """Returns a dict from a YAML string.

    Args:
        yaml_str (str): the string with the YAML contents

    Returns:
        dict: with the yaml content
    """
    import yaml
    d = yaml.load(yaml_str)
    if d is not None:
        return d
    return {}


def yaml_file_to_dict(file_name):
    """Returns a dict from a YAML file.

    Args:
        file_name (str): the path to the the YAML file.

    Returns:
        dict: with the yaml content
    """
    with open(file_name) as f:
        return yaml_string_to_dict(f.read())


def set_data_type(maps_dict, numpy_data_type=np.float32):
    """Convert all maps in the given dictionary to the given numpy data type.

    Args:
        maps_dict (dict): the dictionary with the parameter maps
        numpy_data_type (np datatype): the data type to convert the maps to. Use for example np.float32 for float and
            np.float64 for double.

    Returns:
        the same dictionary with updated maps. This means the conversion happens in-place.
    """
    for k, v in maps_dict.items():
        maps_dict[k] = v.astype(numpy_data_type)
    return maps_dict


def get_config_dir():
    """Get the location of the components.

    Return:
        str: the path to the components
    """
    import os
    return os.path.join(os.path.expanduser("~"), '.mdt', __version__)


def recalculate_ics(model, problem_data, data_dir, sigma, output_dir=None, sigma_param_name=None,
                    evaluation_model=None):
    """Recalculate the information criterion maps.

    This will write the results either to the original data directory, or to the given output dir.

    Args:
        model (str or AbstractModel): An implementation of an AbstractModel that contains the model we want to optimize
            or the name of an model we load with get_model()
        problem_data (DMRIProblemData): the problem data object
        data_dir (str): the directory containing the results for the given model
        sigma (float): the new noise sigma we use for calculating the log likelihood and then the
            information criteria's.
        output_dir (str): if given, we write the output to this directory instead of the data dir.
        sigma_param_name (str): the name of the parameter to which we will set sigma. If not given we search
            the result maps for something ending in .sigma
        evaluation_model: the evaluation model, we will manually fix the sigma in this function
    """
    import mdt.utils
    from mdt.models.cascade import DMRICascadeModelInterface
    from mot.cl_routines.mapping.loglikelihood_calculator import LogLikelihoodCalculator
    from mot.model_building.evaluation_models import OffsetGaussianEvaluationModel

    logger = logging.getLogger(__name__)

    if isinstance(model, string_types):
        model = get_model(model)

    if isinstance(model, DMRICascadeModelInterface):
        raise ValueError('This function does not accept cascade models.')

    model.set_problem_data(problem_data)

    results_maps = create_roi(load_volume_maps(data_dir), problem_data.mask)

    if sigma_param_name is None:
        sigma_params = list(filter(lambda key: '.sigma' in key, model.get_optimization_output_param_names()))

        if not sigma_params:
            raise ValueError('Could not find a suitable parameter to set sigma for.')

        sigma_param_name = sigma_params[0]
        logger.info('Setting the given sigma value to the model parameter {}.'.format(sigma_param_name))

    model.fix(sigma_param_name, sigma)

    evaluation_model = evaluation_model or OffsetGaussianEvaluationModel()
    evaluation_model.set_noise_level_std(sigma, fix=True)

    log_likelihood_calc = LogLikelihoodCalculator()
    log_likelihoods = log_likelihood_calc.calculate(model, results_maps, evaluation_model=evaluation_model)

    k = model.get_nmr_estimable_parameters()
    n = problem_data.protocol.length
    results_maps.update({'LogLikelihood': log_likelihoods})
    results_maps.update(utils.calculate_information_criterions(log_likelihoods, k, n))

    volumes = mdt.utils.restore_volumes(results_maps, problem_data.mask)

    output_dir = output_dir or data_dir
    write_volume_maps(volumes, output_dir, problem_data.volume_header)


def roi_index_to_volume_index(roi_index, brain_mask):
    """Get the 3d index of a voxel given the linear index in a ROI created with the given brain mask.

    This is the inverse function of volume_index_to_roi_index.

    This function is useful if you, for example, have sampling results of a specific voxel
    and you want to locate that voxel in the brain maps.

    Args:
        roi_index (int): the index in the ROI created by that brain mask
        brain_mask (str or 3d array): the brain mask you would like to use

    Returns:
        tuple: the 3d voxel location of the indicated voxel
    """
    from mdt.data_loaders.brain_mask import autodetect_brain_mask_loader
    mask = autodetect_brain_mask_loader(brain_mask).get_data()

    index_matrix = np.indices(mask.shape[0:3])
    index_matrix = np.transpose(index_matrix, (1, 2, 3, 0))

    roi = create_roi(index_matrix, mask)
    return tuple(roi[roi_index])


def volume_index_to_roi_index(volume_index, brain_mask):
    """Get the ROI index given the volume index (in 3d).

    This is the inverse function of roi_index_to_volume_index.

    This function is useful if you want to locate a voxel in the ROI given the position in the volume.

    Args:
        volume_index (tuple): the volume index, a tuple or list of length 3
        brain_mask (str or 3d array): the brain mask you would like to use

    Returns:
        int: the index of the given voxel in the ROI created by the given mask
    """
    return create_index_matrix(brain_mask)[volume_index]


def create_index_matrix(brain_mask):
    """Get a matrix with on every 3d position the linear index number of that voxel.

    This function is useful if you want to locate a voxel in the ROI given the position in the volume.

    Args:
        brain_mask (str or 3d array): the brain mask you would like to use

    Returns:
        3d ndarray: a 3d volume of the same size as the given mask and with as every non-zero element the position
            of that voxel in the linear ROI list.
    """
    from mdt.data_loaders.brain_mask import autodetect_brain_mask_loader
    mask = autodetect_brain_mask_loader(brain_mask).get_data()
    roi_length = np.count_nonzero(mask)
    roi = np.arange(0, roi_length)
    return restore_volumes(roi, mask, with_volume_dim=False)


def build_optimizer(optimizer_info):
    """Build an optimizer either from a YAML string or a dictionary.

    This uses the MetaOptimizerBuilder from mdt.utils to create a optimizer. This optimizer can then be used
    as input to the fit_model routine.

    Args:
        optimizer_info (str or dict): either a YAML string or a dict containing the information about the optimizer
            to build.

    Returns:
        optimizer: a MetaOptimizer with the specific settings
    """
    import six
    import yaml
    from mdt.utils import MetaOptimizerBuilder
    if isinstance(optimizer_info, six.string_types):
        optimizer_info = yaml.load(optimizer_info)
    return MetaOptimizerBuilder(optimizer_info).construct()


def get_data_shape(image_file):
    """Get the data shape of an image on file.

    Args:
        image_file (str): the filename of the image to get the shape of

    Returns:
        tuple: the shape of the image file
    """
    return load_nifti(image_file).get_header().get_data_shape()


try:
    from mdt import configuration
    try:
        conf = configuration.config['logging']['info_dict']
        logging_config.dictConfig(configuration.config['logging']['info_dict'])
    except ValueError:
        print('Logging disabled')
except ImportError:
    # We are probably importing this file in the setup.py for installation.
    pass
