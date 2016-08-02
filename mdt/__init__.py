import glob
import os
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


VERSION = '0.8.7'
VERSION_STATUS = ''

_items = VERSION.split('-')
VERSION_NUMBER_PARTS = tuple(int(i) for i in _items[0].split('.'))
if len(_items) > 1:
    VERSION_STATUS = _items[1]
__version__ = VERSION


try:
    from mdt.configuration import get_logging_configuration_dict
    try:
        logging_config.dictConfig(get_logging_configuration_dict())
    except ValueError:
        print('Logging disabled')
except ImportError:
    # We are probably importing this file in the setup.py for installation.
    pass


from mdt.utils import estimate_noise_std, get_cl_devices, load_problem_data, create_blank_mask, create_index_matrix, \
    volume_index_to_roi_index, roi_index_to_volume_index, load_brain_mask, init_user_settings, restore_volumes, \
    apply_mask, create_roi, volume_merge, concatenate_mri_sets

from mdt.batch_utils import collect_batch_fit_output, run_function_on_batch_fit_output

from mdt.protocols import load_bvec_bval, load_protocol, auto_load_protocol, write_protocol, write_bvec_bval

from mdt.components_loader import load_component, get_model

from mdt.configuration import config_context


def fit_model(model, problem_data, output_folder, optimizer=None,
              recalculate=False, only_recalculate_last=False, cascade_subdir=False,
              cl_device_ind=None, double_precision=False, tmp_results_dir=True):
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
        cascade_subdir (boolean): if we want to create a subdirectory for the given model if it is a cascade model.
            Per default we output the maps of cascaded results in the same directory, this allows reusing cascaded
            results for other cascades (for example, if you cascade BallStick -> Noddi you can use the BallStick results
            also for BallStick -> Charmed). This flag disables that behaviour and instead outputs the results of
            a cascade model to a subdirectory for that cascade. This does not apply recursive.
        cl_device_ind (int or list): the index of the CL device to use. The index is from the list from the function
            utils.get_cl_devices(). This can also be a list of device indices.
        double_precision (boolean): if we would like to do the calculations in double precision
        tmp_results_dir (str, True or None): The temporary dir for the calculations. Set to a string to use
                that path directly, set to True to use the config value, set to None to disable.

    Returns:
        the output of the optimization. If a cascade is given, only the results of the last model in the cascade is
        returned
    """
    import mdt.utils
    from mdt.model_fitting import ModelFit

    if not mdt.utils.check_user_components():
        raise RuntimeError('Your components folder is not up to date. Please run the script mdt-init-user-settings.')

    model_fit = ModelFit(model, problem_data, output_folder, optimizer=optimizer, recalculate=recalculate,
                         only_recalculate_last=only_recalculate_last,
                         cascade_subdir=cascade_subdir,
                         cl_device_ind=cl_device_ind, double_precision=double_precision,
                         tmp_results_dir=tmp_results_dir)

    return model_fit.run()


def sample_model(model, problem_data, output_folder, sampler=None, recalculate=False,
                 cl_device_ind=None, double_precision=False, initialize=True, initialize_using=None,
                 store_samples=True, tmp_results_dir=True):
    """Sample a single model. This does not accept cascade models, only single models.

    Args:
        model: the model to sample
        problem_data (DMRIProblemData): the problem data object, load with, for example, mdt.load_problem_data().
        output_folder (string): The path to the folder where to place the output, we will make a subdir with the
            model name in it (for the optimization results) and then a subdir with the samples output.
        sampler (AbstractSampler): the sampler to use
        recalculate (boolean): If we want to recalculate the results if they are already present.
        cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
            utils.get_cl_devices().
        double_precision (boolean): if we would like to do the calculations in double precision
        initialize (boolean): If we want to initialize the sampler with optimization output.
            This assumes that the optimization results are in the folder:
                <output_folder>/<model_name>/
        initialize_using (None, str, or dict): If None, and initialize is True we will initialize from the
            optimization maps from a model with the same name. If a string is given and initialize is True we will
            interpret the string as a folder with the maps to load. If a dict is given and initialize is True we will
            initialize from the dict directly.
        store_samples (boolean): if set to False we will store none of the samples. Use this
                if you are only interested in the volume maps and not in the entire sample chain.
        tmp_results_dir (str, True or None): The temporary dir for the calculations. Set to a string to use
                that path directly, set to True to use the config value, set to None to disable.

    Returns:
        dict: the samples per parameter as a numpy memmap, if store_samples is True
    """
    import mdt.utils
    from mdt.model_sampling import ModelSampling

    if not mdt.utils.check_user_components():
        raise RuntimeError('Your components folder is not up to date. Please run the script mdt-init-user-settings.')

    sampling = ModelSampling(model, problem_data, output_folder,
                             sampler=sampler, recalculate=recalculate, cl_device_ind=cl_device_ind,
                             double_precision=double_precision,
                             initialize=initialize,
                             initialize_using=initialize_using, store_samples=store_samples,
                             tmp_results_dir=tmp_results_dir)

    return sampling.run()


def batch_fit(data_folder, batch_profile=None, subjects_selection=None, recalculate=False,
              models_to_fit=None, cascade_subdir=False, cl_device_ind=None, dry_run=False,
              double_precision=False, tmp_results_dir=True):
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
        tmp_results_dir (str, True or None): The temporary dir for the calculations. Set to a string to use
                that path directly, set to True to use the config value, set to None to disable.

    Returns:
        The list of subjects we will calculate / have calculated.
    """
    import mdt.utils
    from mdt.model_fitting import BatchFitting

    if not mdt.utils.check_user_components():
        raise RuntimeError('Your components folder is not up to date. Please run the script mdt-init-user-settings.')

    batch_fitting = BatchFitting(data_folder, batch_profile=batch_profile, subjects_selection=subjects_selection,
                                 recalculate=recalculate, models_to_fit=models_to_fit, cascade_subdir=cascade_subdir,
                                 cl_device_ind=cl_device_ind, double_precision=double_precision,
                                 tmp_results_dir=tmp_results_dir)

    if dry_run:
        return batch_fitting.get_subjects_info()

    return batch_fitting.run()


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

    filter_match = ('.vec', '.d', '.sigma', 'AIC', 'Errors.mse', 'Errors.sse', '.eigen_ranking',
                    'SignalEstimates', 'UsedMask')
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

    if isinstance(data, string_types):
        data = load_samples(data)

    if not kwargs.get('voxel_ind'):
        kwargs.update({'voxel_ind': data[list(data.keys())[0]].shape[0] / 2})
    SampleVisualizer(data).show(**kwargs)


def load_samples(data_folder, mode='r'):
    """Load sampled results as a dictionary of numpy memmap.

    Args:
        data_folder (str): the folder from which to load the samples
        mode (str): the mode in which to open the memory mapped sample files (see numpy mode parameter)

    Returns:
        dict: the memory loaded samples per sampled parameter.
    """
    from mdt.utils import load_samples
    return load_samples(data_folder, mode=mode)


def load_nifti(nifti_volume):
    """Load and return a nifti file.

    This will apply path resolution if a filename without extension is given. See the function
    mdt.utils.nifti_filepath_resolution() for details.

    Args:
        nifti_volume (string): The filename of the volume to load.

    Returns:
        nib image proxy (from nib.load)
    """
    from mdt.utils import load_nifti
    return load_nifti(nifti_volume)


def make_path_joiner(*folder):
    """Generates and returns an instance of utils.PathJoiner to quickly join pathnames.

    Returns:
        An instance of utils.PathJoiner for easy path manipulation.
    """
    from mdt.utils import PathJoiner
    return PathJoiner(*folder)


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
    from mdt.utils import create_slice_roi

    if os.path.exists(output_fname) and not overwrite_if_exists:
        return load_brain_mask(output_fname)

    if not os.path.isdir(os.path.dirname(output_fname)):
        os.makedirs(os.path.dirname(output_fname))

    brain_mask_img = load_nifti(brain_mask_fname)
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
                tmp.append(load_nifti(data).get_data())
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
        sort_index_map = np.round(load_nifti(sort_index_map).get_data()).astype(np.int64)

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


def get_batch_profile(batch_profile_name, *args, **kwargs):
    """Load one of the batch profiles.

    This is short for load_component('batch_profiles', batch_profile_name).

    Args:
        batch_profile_name (str): the name of the batch profile to load

    Returns:
        BatchProfile: the batch profile for use in batch fitting routines.
    """
    return load_component('batch_profiles', batch_profile_name, *args, **kwargs)


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
    import mdt.utils
    from mdt.IO import Nifti

    if output_folder is None:
        output_folder = os.path.dirname(input_fname)
    dataset = load_nifti(input_fname)
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

    input_volume = load_nifti(input_volume_fname)
    image_data = input_volume.get_data()[..., protocol_indices]
    write_image(output_volume_fname, image_data, input_volume.get_header())


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

    write_image(output_fname, apply_mask(input_fname, mask), load_nifti(input_fname).get_header())


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
    n = problem_data.get_nmr_inst_per_problem()
    results_maps.update({'LogLikelihood': log_likelihoods})
    results_maps.update(utils.calculate_information_criterions(log_likelihoods, k, n))

    volumes = mdt.utils.restore_volumes(results_maps, problem_data.mask)

    output_dir = output_dir or data_dir
    write_volume_maps(volumes, output_dir, problem_data.volume_header)
