import glob
import logging.config as logging_config
import os
from inspect import stack
import numpy as np
import six
from six import string_types

__author__ = 'Robbert Harms'
__date__ = "2015-03-10"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


VERSION = '0.8.12'
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


from mdt.user_script_info import easy_save_user_script_info
from mdt.utils import estimate_noise_std, get_cl_devices, load_problem_data, create_blank_mask, create_index_matrix, \
    volume_index_to_roi_index, roi_index_to_volume_index, load_brain_mask, init_user_settings, restore_volumes, \
    apply_mask, create_roi, volume_merge, concatenate_mri_sets, create_median_otsu_brain_mask, load_samples, \
    load_nifti, write_slice_roi, split_write_dataset, apply_mask_to_file, extract_volumes, recalculate_error_measures, \
    create_signal_estimates, get_slice_in_dimension
from mdt.batch_utils import collect_batch_fit_output, run_function_on_batch_fit_output
from mdt.protocols import load_bvec_bval, load_protocol, auto_load_protocol, write_protocol, write_bvec_bval
from mdt.components_loader import load_component, get_model
from mdt.configuration import config_context


def fit_model(model, problem_data, output_folder, optimizer=None,
              recalculate=False, only_recalculate_last=False, cascade_subdir=False,
              cl_device_ind=None, double_precision=False, tmp_results_dir=True, save_user_script_info=True):
    """Run the optimizer on the given model.

    Args:
        model (str or AbstractModel): An implementation of an AbstractModel that contains the model we want to optimize
            or the name of an model we set_current_map with get_model()
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
        save_user_script_info (boolean, str or SaveUserScriptInfo): The info we need to save about the script the
            user is currently executing. If True (default) we use the stack to lookup the script the user is executing
            and save that using a SaveFromScript saver. If a string is given we use that filename again for the
            SaveFromScript saver. If False or None, we do not write any information. If a SaveUserScriptInfo is
            given we use that directly.

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

    results = model_fit.run()
    easy_save_user_script_info(save_user_script_info, output_folder + '/used_scripts.py',
                               stack()[1][0].f_globals.get('__file__'))
    return results


def sample_model(model, problem_data, output_folder, sampler=None, recalculate=False,
                 cl_device_ind=None, double_precision=False, initialize=True, initialize_using=None,
                 store_samples=True, tmp_results_dir=True, save_user_script_info=True):
    """Sample a single model. This does not accept cascade models, only single models.

    Args:
        model: the model to sample
        problem_data (DMRIProblemData): the problem data object, set_current_map with, for example, mdt.load_problem_data().
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
            interpret the string as a folder with the maps to set_current_map. If a dict is given and initialize is True we will
            initialize from the dict directly.
        store_samples (boolean): if set to False we will store none of the samples. Use this
                if you are only interested in the volume maps and not in the entire sample chain.
        tmp_results_dir (str, True or None): The temporary dir for the calculations. Set to a string to use
                that path directly, set to True to use the config value, set to None to disable.
        save_user_script_info (boolean, str or SaveUserScriptInfo): The info we need to save about the script the
            user is currently executing. If True (default) we use the stack to lookup the script the user is executing
            and save that using a SaveFromScript saver. If a string is given we use that filename again for the
            SaveFromScript saver. If False or None, we do not write any information. If a SaveUserScriptInfo is
            given we use that directly.

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

    results = sampling.run()
    easy_save_user_script_info(save_user_script_info, output_folder + '/used_scripts.py',
                               stack()[1][0].f_globals.get('__file__'))
    return results


def batch_fit(data_folder, batch_profile=None, subjects_selection=None, recalculate=False,
              models_to_fit=None, cascade_subdir=False, cl_device_ind=None, dry_run=False,
              double_precision=False, tmp_results_dir=True):
    """Run all the available and applicable models on the data in the given folder.

    See the class AutoRun for more details and options.

    Setting the cl_device_ind has the side effect that it changes the current run time cl_device settings in the MOT
    toolkit.

    Args:
        data_folder (str): The data folder to process
        batch_profile (BatchProfile or str): the batch profile to use, or the name of a batch profile to set_current_map.
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


def view_maps(data, config=None, to_file=None, to_file_options=None,
              block=True, show_maximized=False, use_qt=True, figure_options=None):
    """View a number of maps using the MDT Maps Visualizer.

    Args:
        data (str, dict, DataInfo): the data we are showing, either a dictionary with result maps, a string with
            a path name or a DataInfo object
        config (str, dict, MapPlotConfig): either a Yaml string or a dictionary with configuration settings or a
            MapPlotConfig object to use directly
        to_file (str): if set we output the figure to a file and do not launch a GUI
        to_file_options (dict): extra output options for the savefig command from matplotlib
        block (boolean): if we block the plots or not
        show_maximized (boolean): if we show the window maximized or not
        use_qt (boolean): if we want to use the Qt GUI, or show the results directly in matplotlib
        figure_options (dict): figure options for matplotlib.Figure
    """
    from mdt.gui.maps_visualizer.main import start_gui
    from mdt.gui.maps_visualizer.base import ValidatedMapPlotConfig
    from mdt.visualization.maps.matplotlib_renderer import MapsVisualizer
    import matplotlib.pyplot as plt
    from mdt.visualization.maps.base import DataInfo

    if isinstance(data, string_types):
        data = DataInfo.from_dir(data)
    elif isinstance(data, dict):
        data = DataInfo(data)
    elif data is None:
        data = DataInfo({})

    if config is None:
        config = ValidatedMapPlotConfig()
    elif isinstance(config, string_types):
        config = ValidatedMapPlotConfig.from_yaml(config)
    elif isinstance(config, dict):
        config = ValidatedMapPlotConfig.from_dict(config)

    if to_file:
        figure_options = figure_options or {}
        figure_options['figsize'] = figure_options.get('figsize', (18, 16))
        figure_options['dpi'] = figure_options.get('dpi', 100)

        figure = plt.figure(**figure_options)
        viz = MapsVisualizer(data, figure)

        to_file_options = to_file_options or {}

        viz.to_file(to_file, config, **to_file_options)
    elif use_qt:
        start_gui(data, config, app_exec=block, show_maximized=show_maximized)
    else:
        figure_options = figure_options or {}
        figure_options['figsize'] = figure_options.get('figsize', (18, 16))
        figure_options['dpi'] = figure_options.get('dpi', 100)

        figure = plt.figure(**figure_options)
        viz = MapsVisualizer(data, figure)
        viz.show(config, block=block, maximize=show_maximized)


def results_preselection_names(data):
    """Generate a list of useful map names to display.

    This is primarily to be used as argument to the parameter 'maps_to_show' of the function view_maps.

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


def block_plots(use_qt=True):
    """A small function to block matplotlib plots and Qt GUI instances.

    This basically calls either plt.show() and QtApplication.exec_() depending on use_qt.

    Args:
        use_qt (boolean): if True we block Qt windows, if False we block matplotlib windows
    """
    if use_qt:
        from mdt.gui.utils import QtManager
        QtManager.exec_()
    else:
        import matplotlib.pyplot as plt
        plt.show()


def view_result_samples(data, **kwargs):
    """View the samples from the given results set.

    Args:
        data (string or dict): The location of the maps to set_current_map the samples from, or the samples themselves.
        kwargs (dict): see SampleVisualizer for all the supported keywords
    """
    from mdt.visualization.samples import SampleVisualizer

    if isinstance(data, string_types):
        data = load_samples(data)

    if not kwargs.get('voxel_ind'):
        kwargs.update({'voxel_ind': data[list(data.keys())[0]].shape[0] / 2})
    SampleVisualizer(data).show(**kwargs)


def make_path_joiner(*folder):
    """Generates and returns an instance of utils.PathJoiner to quickly join pathnames.

    Returns:
        An instance of utils.PathJoiner for easy path manipulation.
    """
    from mdt.utils import PathJoiner
    return PathJoiner(*folder)


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
        maps_to_sort_on (list): a list of string (filenames) or ndarrays we will set_current_map and compare
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
        map_names: the names of the maps we want to set_current_map. If given we only set_current_map and return these maps.

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
        batch_profile_name (str): the name of the batch profile to set_current_map

    Returns:
        BatchProfile: the batch profile for use in batch fitting routines.
    """
    return load_component('batch_profiles', batch_profile_name, *args, **kwargs)
