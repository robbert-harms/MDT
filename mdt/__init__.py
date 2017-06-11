import collections
import glob
import logging
import logging.config as logging_config
import os
from inspect import stack
from contextlib import contextmanager
import numpy as np
import shutil
import six
from six import string_types

from .__version__ import VERSION, VERSION_STATUS, __version__


from mdt.configuration import get_logging_configuration_dict
try:
    logging_config.dictConfig(get_logging_configuration_dict())
except ValueError:
    print('Logging disabled')


from mdt.user_script_info import easy_save_user_script_info
from mdt.utils import estimate_noise_std, get_cl_devices, load_problem_data, create_blank_mask, create_index_matrix, \
    volume_index_to_roi_index, roi_index_to_volume_index, load_brain_mask, init_user_settings, restore_volumes, \
    apply_mask, create_roi, volume_merge, protocol_merge, create_median_otsu_brain_mask, load_samples, \
    load_nifti, write_slice_roi, split_write_dataset, apply_mask_to_file, extract_volumes, recalculate_error_measures, \
    get_slice_in_dimension, per_model_logging_context, \
    get_temporary_results_dir, get_example_data, create_sort_matrix, sort_volumes_per_voxel, \
    sort_orientations, SimpleInitializationData
from mdt.simulations import create_signal_estimates, simulate_signals, add_rician_noise
from mdt.batch_utils import collect_batch_fit_output, collect_batch_fit_single_map, run_function_on_batch_fit_output
from mdt.protocols import load_bvec_bval, load_protocol, auto_load_protocol, write_protocol, write_bvec_bval
from mdt.components_loader import load_component, get_model, component_import, construct_component, get_component_class
from mdt.configuration import config_context, get_processing_strategy
from mdt.exceptions import InsufficientProtocolError
from mdt.nifti import write_nifti


__author__ = 'Robbert Harms'
__date__ = "2015-03-10"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def fit_model(model, problem_data, output_folder, optimizer=None,
              recalculate=False, only_recalculate_last=False, cascade_subdir=False,
              cl_device_ind=None, double_precision=False, tmp_results_dir=True, save_user_script_info=True,
              initialization_data=None):
    """Run the optimizer on the given model.

    Args:
        model (str or :class:`~mdt.models.composite.DMRICompositeModel` or :class:`~mdt.models.cascade.DMRICascadeModelInterface`):
            An implementation of an AbstractModel that contains the model we want to optimize or the name of
            an model.
        problem_data (:class:`~mdt.utils.DMRIProblemData`): the problem data object containing all the info needed for
            diffusion MRI model fitting
        output_folder (string): The path to the folder where to place the output, we will make a subdir with the
            model name in it.
        optimizer (:class:`mot.cl_routines.optimizing.base.AbstractOptimizer`): The optimization routine to use.
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
        initialization_data (:class:`~mdt.utils.InitializationData`): provides (extra) initialization data to use
            during model fitting. If we are optimizing a cascade model this data only applies to the last model in the
            cascade.

    Returns:
        dict: The result maps for the given composite model or the last model in the cascade.
            This returns the results as 3d/4d volumes for every output map.
    """
    import mdt.utils
    from mdt.model_fitting import ModelFit

    if not mdt.utils.check_user_components():
        init_user_settings(pass_if_exists=True)

    model_fit = ModelFit(model, problem_data, output_folder, optimizer=optimizer, recalculate=recalculate,
                         only_recalculate_last=only_recalculate_last,
                         cascade_subdir=cascade_subdir,
                         cl_device_ind=cl_device_ind, double_precision=double_precision,
                         tmp_results_dir=tmp_results_dir, initialization_data=initialization_data)

    results = model_fit.run()
    easy_save_user_script_info(save_user_script_info, output_folder + '/used_scripts.py',
                               stack()[1][0].f_globals.get('__file__'))
    return results


def sample_model(model, problem_data, output_folder, sampler=None, recalculate=False,
                 cl_device_ind=None, double_precision=False, store_samples=True,
                 tmp_results_dir=True, save_user_script_info=True, initialization_data=None):
    """Sample a composite model using the given cascading strategy.

    Args:
        model (:class:`~mdt.models.composite.DMRICompositeModel` or str): the model to sample
        problem_data (:class:`~mdt.utils.DMRIProblemData`): the problem data object
        output_folder (string): The path to the folder where to place the output, we will make a subdir with the
            model name in it (for the optimization results) and then a subdir with the samples output.
        sampler (:class:`mot.cl_routines.sampling.base.AbstractSampler`): the sampler to use
        recalculate (boolean): If we want to recalculate the results if they are already present.
        cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
            utils.get_cl_devices().
        double_precision (boolean): if we would like to do the calculations in double precision
        store_samples (boolean or int): if set to False we will store none of the samples. If set to an integer we will
            store only thinned samples with that amount.
        tmp_results_dir (str, True or None): The temporary dir for the calculations. Set to a string to use
                that path directly, set to True to use the config value, set to None to disable.
        save_user_script_info (boolean, str or SaveUserScriptInfo): The info we need to save about the script the
            user is currently executing. If True (default) we use the stack to lookup the script the user is executing
            and save that using a SaveFromScript saver. If a string is given we use that filename again for the
            SaveFromScript saver. If False or None, we do not write any information. If a SaveUserScriptInfo is
            given we use that directly.
        initialization_data (:class:`~mdt.utils.InitializationData`): provides (extra) initialization data to use
            during model fitting. If we are optimizing a cascade model this data only applies to the last model in the
            cascade.

    Returns:
        dict: if store_samples is True then we return the samples per parameter as a numpy memmap. If store_samples
            is False we return None
    """
    import mdt.utils
    from mot.load_balance_strategies import EvenDistribution
    from mdt.model_sampling import sample_composite_model
    from mdt.models.cascade import DMRICascadeModelInterface
    import mot.configuration

    if not mdt.utils.check_user_components():
        init_user_settings(pass_if_exists=True)

    if isinstance(model, string_types):
        model = get_model(model)

    if isinstance(model, DMRICascadeModelInterface):
        raise ValueError('The function \'sample_model()\' does not accept cascade models.')

    if not model.is_protocol_sufficient(problem_data.protocol):
        raise InsufficientProtocolError(
            'The given protocol is insufficient for this model. '
            'The reported errors where: {}'.format(model.get_protocol_problems(problem_data.protocol)))

    if cl_device_ind is not None and not isinstance(cl_device_ind, collections.Iterable):
        cl_device_ind = [cl_device_ind]

    if cl_device_ind is None:
        cl_context_action = mot.configuration.VoidConfigurationAction()
    else:
        cl_context_action = mot.configuration.RuntimeConfigurationAction(
            cl_environments=[get_cl_devices()[ind] for ind in cl_device_ind],
            load_balancer=EvenDistribution())

    with mot.configuration.config_context(cl_context_action):
        if sampler is None:
            sampler = configuration.get_sampler()

        processing_strategy = get_processing_strategy('sampling', model_names=model.name,
                                                      tmp_dir=get_temporary_results_dir(tmp_results_dir))

        base_dir = os.path.join(output_folder, model.name, 'samples')
        output_folder = base_dir

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        if recalculate:
            shutil.rmtree(output_folder)

        with per_model_logging_context(base_dir, overwrite=recalculate):
            logger = logging.getLogger(__name__)
            logger.info('Using MDT version {}'.format(__version__))
            logger.info('Preparing for model {0}'.format(model.name))

            model.double_precision = double_precision

            results = sample_composite_model(model, problem_data, output_folder, sampler,
                                             processing_strategy, recalculate=recalculate,
                                             store_samples=store_samples,
                                             initialization_data=initialization_data)

        easy_save_user_script_info(save_user_script_info, os.path.join(base_dir, 'used_scripts.py'),
                                   stack()[1][0].f_globals.get('__file__'))
        return results


def batch_fit(data_folder, models_to_fit, batch_profile=None, subjects_selection=None, recalculate=False,
              cascade_subdir=False, cl_device_ind=None, dry_run=False,
              double_precision=False, tmp_results_dir=True):
    """Run all the available and applicable models on the data in the given folder.

    Args:
        data_folder (str): The data folder to process
        models_to_fit (list of str): A list of models to fit to the data.
        batch_profile (:class:`~mdt.batch_utils.BatchProfile` or str): the batch profile to use,
            or the name of a batch profile to use. If not given it is auto detected.
        subjects_selection (:class:`~mdt.batch_utils.BatchSubjectSelection`): the subjects to use for processing.
            If None all subjects are processed.
        recalculate (boolean): If we want to recalculate the results if they are already present.
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
        init_user_settings(pass_if_exists=True)

    batch_fitting = BatchFitting(data_folder, models_to_fit, batch_profile=batch_profile,
                                 subjects_selection=subjects_selection,
                                 recalculate=recalculate, cascade_subdir=cascade_subdir,
                                 cl_device_ind=cl_device_ind, double_precision=double_precision,
                                 tmp_results_dir=tmp_results_dir)

    if dry_run:
        return batch_fitting.get_subjects_info()

    return batch_fitting.run()


def view_maps(data, config=None, figure_options=None,
              block=True, show_maximized=False, use_qt=True,
              window_title=None, enable_directory_watcher=True):
    """View a number of maps using the MDT Maps Visualizer.

    Args:
        data (str, dict, :class:`~mdt.visualization.maps.base.DataInfo`): the data we are showing,
            either a dictionary with result maps, a string with a path name or a DataInfo object
        config (str, dict, :class:`~mdt.visualization.maps.base import MapPlotConfig`): either a Yaml string or a
            dictionary with configuration settings or a ValidatedMapPlotConfig object to use directly
        figure_options (dict): figure options for the matplotlib Figure, if figsizes is not given you can also specify
            two ints, width and height, to indicate the pixel size of the resulting figure, together with the dpi they
            are used to calculate the figsize. Only used if use_qt=False.
        block (boolean): if we block the plots or not
        show_maximized (boolean): if we show the window maximized or not
        window_title (str): the title for the window
        use_qt (boolean): if we want to use the Qt GUI, or show the results directly in matplotlib
        enable_directory_watcher (boolean): if the directory watcher should be enabled/disabled, only applicable for the
            QT GUI. If the directory watcher is enabled, the viewer will automatically add new maps when added
            to the folder and also automatically remove maps when they are removed from the directory.
            It is useful to disable this if you want to have multiple viewers open with old results.
    """
    from mdt.gui.maps_visualizer.main import start_gui
    from mdt.visualization.maps.base import MapPlotConfig
    from mdt.visualization.maps.matplotlib_renderer import MapsVisualizer
    import matplotlib.pyplot as plt
    from mdt.visualization.maps.base import SimpleDataInfo

    if isinstance(data, string_types):
        data = SimpleDataInfo.from_dir(data)
    elif isinstance(data, collections.MutableMapping):
        data = SimpleDataInfo(data)
    elif data is None:
        data = SimpleDataInfo({})

    if config is None:
        config = MapPlotConfig()
    elif isinstance(config, string_types):
        if config.strip():
            config = MapPlotConfig.from_yaml(config)
        else:
            config = MapPlotConfig()
    elif isinstance(config, dict):
        config = MapPlotConfig.from_dict(config)

    if use_qt:
        start_gui(data, config, app_exec=block, show_maximized=show_maximized, window_title=window_title,
                  enable_directory_watcher=enable_directory_watcher)
    else:
        figure_options = figure_options or {}
        figure_options['dpi'] = figure_options.get('dpi', 100)
        if 'figsize' not in figure_options:
            figure_options['figsize'] = (figure_options.pop('width', 1800) / figure_options['dpi'],
                                         figure_options.pop('height', 1600) / figure_options['dpi'])

        figure = plt.figure(**figure_options)
        viz = MapsVisualizer(data, figure)
        viz.show(config, block=block, maximize=show_maximized)


def write_view_maps_figure(data, output_filename, config=None, width=None, height=None, dpi=None,
                           figure_options=None, savefig_settings=None):
    """Saves the view maps figure to a file.

    Args:
        data (str, dict, :class:`~mdt.visualization.maps.base.DataInfo`): the data we are showing,
            either a dictionary with result maps, a string with a path name or a DataInfo object
        config (str, dict, :class:`~mdt.visualization.maps.base import MapPlotConfig`): either a Yaml string or a
            dictionary with configuration settings or a ValidatedMapPlotConfig object to use directly
        output_filename (str): the output filename
        width (int): the width of the figure, if set it takes precedence over the value in figure_options
        height (int): the height of the figure, if set it takes precedence over the value in figure_options
        dpi (int): the dpi of the figure, if set it takes precedence over the value in figure_options
        figure_options (dict): additional figure options for the matplotlib Figure, supports the
            same settings as :func:`view_maps`, that is, if the arguments 'width' and 'height' are not set directly
            in this function we can also use those in the figure_options
        savefig_settings (dict): extra output options for the savefig command from matplotlib, if dpi is not given, we
            use the dpi from the figure_options.
    """
    from mdt.gui.maps_visualizer.main import start_gui
    from mdt.visualization.maps.base import MapPlotConfig
    from mdt.visualization.maps.matplotlib_renderer import MapsVisualizer
    import matplotlib.pyplot as plt
    from mdt.visualization.maps.base import SimpleDataInfo

    if isinstance(data, string_types):
        data = SimpleDataInfo.from_dir(data)
    elif isinstance(data, dict):
        data = SimpleDataInfo(data)
    elif data is None:
        data = SimpleDataInfo({})

    if config is None:
        config = MapPlotConfig()
    elif isinstance(config, string_types):
        if config.strip():
            config = MapPlotConfig.from_yaml(config)
        else:
            config = MapPlotConfig()
    elif isinstance(config, dict):
        config = MapPlotConfig.from_dict(config)

    figure_options = figure_options or {}

    if dpi is not None:
        figure_options['dpi'] = dpi
    else:
        figure_options['dpi'] = figure_options.get('dpi', 80)

    if height is not None and width is not None:
        figure_options['figsize'] = (width / figure_options['dpi'], height / figure_options['dpi'])
    elif height is not None and width is None:
        width = figure_options.get('width', 800)
        figure_options['figsize'] = (width / figure_options['dpi'], height / figure_options['dpi'])
    elif width is not None and height is None:
        height = figure_options.get('height', 640)
        figure_options['figsize'] = (width / figure_options['dpi'], height / figure_options['dpi'])
    elif 'figsize' in figure_options:
        pass
    else:
        width = figure_options.get('width', 800)
        height = figure_options.get('height', 640)
        figure_options['figsize'] = (width / figure_options['dpi'], height / figure_options['dpi'])

    if 'width' in figure_options:
        del figure_options['width']
    if 'height' in figure_options:
        del figure_options['height']

    figure = plt.figure(**figure_options)
    viz = MapsVisualizer(data, figure)

    savefig_settings = savefig_settings or {}
    savefig_settings['dpi'] = savefig_settings.get('dpi', figure_options['dpi'])
    viz.to_file(output_filename, config, **savefig_settings)


def results_preselection_names(data):
    """Generate a list of useful map names to display.

    This is primarily to be used as argument to the config option ``maps_to_show`` in the function :func:`view_maps`.

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

    This basically calls either ``plt.show()`` and ``QtApplication.exec_()`` depending on ``use_qt``.

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
        data (string or dict): The location of the maps to use the samples from, or the samples themselves.
        kwargs (dict): see SampleVisualizer for all the supported keywords
    """
    from mdt.visualization.samples import SampleVisualizer

    if isinstance(data, string_types):
        data = load_samples(data)

    if kwargs.get('voxel_ind') is None:
        kwargs.update({'voxel_ind': data[list(data.keys())[0]].shape[0] / 2})
    SampleVisualizer(data).show(**kwargs)


def make_path_joiner(*folder):
    """Generates and returns an instance of utils.PathJoiner to quickly join pathnames.

    Returns:
         mdt.utils.PathJoiner: easy path manipulation path joiner
    """
    from mdt.utils import PathJoiner
    return PathJoiner(*folder)


def auto_convert_to_trackmark(input_folder, model_name=None, output_folder=None):
    """Convert the nifti files in the given folder to Trackmark.

    This automatically loads the correct files based on the model name. This is normally the dirname of the given
    path. If that is not the case you can give the model name explicitly.

    By default it outputs the results to a folder named "trackmark" in the given input folder. This can of course
    be overridden using the output_folder parameter.

    Args:
        input_folder (str): the name of the input folder
        model_name (str): the name of the model, if not given we use the last dirname of the given path
        output_folder (str): the output folder, if not given we will output to a subfolder "trackmark" in the
            given directory.
    """
    from mdt.file_conversions.trackmark import TrackMark
    TrackMark.auto_convert(input_folder, model_name=model_name, output_folder=output_folder)


def write_trackmark_rawmaps(data, output_folder, maps_to_convert=None):
    """Convert the given nifti files in the input folder to rawmaps in the output folder.

    Args:
        data (str or dict): the name of the input folder, of a dictionary with maps to save.
        output_folder (str): the name of the output folder. Defaults to <input_folder>/trackmark.
        maps_to_convert (:class:`list`): the list with the names of the maps we want to convert (without the extension).
    """
    from mdt.file_conversions.trackmark import TrackMark

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
    from mdt.file_conversions.trackmark import TrackMark
    if len(vector_directions) != len(vector_magnitudes):
        raise ValueError('The length of the list of vector directions does not '
                         'match with the length of the list of vector magnitudes.')
    TrackMark.write_tvl_direction_pairs(output_tvl, tvl_header, list(zip(vector_directions, vector_magnitudes))[:3])


def sort_maps(input_maps, reversed_sort=False, sort_index_matrix=None):
    """Sort the given maps.

    This first creates a sort matrix to index the maps in sorted order per voxel. Next, it creates the output
    maps for the maps we sort on.

    Args:
        input_maps (:class:`list`): a list of string (filenames) or ndarrays we will sort
        reversed_sort (boolean): if we want to sort from large to small instead of small to large.
            This is not used if a sort index map is provided.
        sort_index_matrix (ndarray): if given we use this sort index map instead of generating one by sorting the
            maps_to_sort_on.

    Returns:
        list: the list of sorted volumes
    """
    if sort_index_matrix is None:
        sort_index_matrix = create_sort_matrix(input_maps, reversed_sort=reversed_sort)
    elif isinstance(sort_index_matrix, string_types):
        sort_index_matrix = np.round(load_nifti(sort_index_matrix).get_data()).astype(np.int64)
    return sort_volumes_per_voxel(input_maps, sort_index_matrix)


def load_volume_maps(directory, map_names=None, deferred=True):
    """Read a number of Nifti volume maps from a directory.

    Args:
        directory (str): the directory from which we want to read a number of maps
        map_names (list or tuple): the names of the maps we want to use. If given we only use and return these maps.
        deferred (boolean): if True we return an deferred loading dictionary instead of a dictionary with the values
            loaded as arrays.

    Returns:
        dict: A dictionary with the volumes. The keys of the dictionary are the filenames (without the extension) of the
            files in the given directory.
    """
    from mdt.nifti import get_all_image_data
    return get_all_image_data(directory, map_names=map_names, deferred=deferred)


def get_volume_names(directory):
    """Get the names of the Nifti volume maps in the given directory.

    Args:
        directory: the directory to get the names of the available maps from.

    Returns:
        :class:`list`: A list with the names of the volumes.
    """
    from mdt.nifti import yield_nifti_info
    return list(sorted(el[1] for el in yield_nifti_info(directory)))


def write_volume_maps(maps, directory, header, overwrite_volumes=True):
    """Write a dictionary with maps to the given directory using the given header.

    Args:
        maps (dict): The maps with as keys the map names and as values 3d or 4d maps
        directory (str): The dir to write to
        header: The Nibabel Image Header
        overwrite_volumes (boolean): If we want to overwrite the volumes if they are present.
    """
    from mdt.nifti import write_all_as_nifti
    write_all_as_nifti(maps, directory, header, overwrite_volumes=overwrite_volumes)


def get_list_of_composite_models():
    """Get a list of all available composite models

    Returns:
        list of str: A list of all available composite model names.
    """
    from mdt.components_loader import CompositeModelsLoader
    return CompositeModelsLoader().list_all()


def get_list_of_cascade_models():
    """Get a list of all available cascade models

    Returns:
        list of str: A list of available cascade models
    """
    from mdt.components_loader import CascadeModelsLoader
    return CascadeModelsLoader().list_all()


def get_models_list():
    """Get a list of all available models, composite and cascade.

    Returns:
        list of str: A list of available model names.
    """
    l = get_list_of_cascade_models()
    l.extend(get_list_of_composite_models())
    return list(sorted(l))


def get_models_meta_info():
    """Get the meta information tags for all the models returned by get_models_list()

    Returns:
        dict of dict: The first dictionary indexes the model names to the meta tags, the second holds the meta
            information.
    """
    from mdt.components_loader import CompositeModelsLoader, CascadeModelsLoader
    sml = CompositeModelsLoader()
    cml = CascadeModelsLoader()

    meta_info = {}
    for model_loader in (sml, cml):
        models = model_loader.list_all()
        for model in models:
            meta_info.update({model: model_loader.get_meta_info(model)})
    return meta_info


def get_batch_profile(batch_profile_name):
    """Load the class reference to one of the batch profiles.

    This is short for mdt.components_loader.get_component_class('batch_profiles', batch_profile_name)

    Args:
        batch_profile_name (str): the name of the batch profile to use

    Returns:
        class: a reference to a :class:`mdt.batch_utils.BatchProfile`.
    """
    return get_component_class('batch_profiles', batch_profile_name)


def gui(base_dir=None, app_exec=True):
    """Start the model fitting GUI.

    Args:
        base_dir (str): the starting directory for the file opening actions
        app_exec (boolean): if true we execute the Qt application, set to false to disable.
            This is only important if you want to start this GUI from within an existing Qt application. If you
            leave this at true in that case, this will try to start a new Qt application which may create problems.
    """
    from mdt.gui.model_fit.qt_main import start_gui
    return start_gui(base_dir=base_dir, app_exec=app_exec)


def reset_logging():
    """Reset the logging to reflect the current configuration.

    This is commonly called after updating the logging configuration to let the changes take affect.
    """
    logging_config.dictConfig(get_logging_configuration_dict())


@contextmanager
def disable_logging_context():
    """A context in which the logging is temporarily disabled"""
    config = '''
    logging:
        info_dict:
            version: 1
            loggers:
                mot:
                    handlers: []

                mdt:
                    handlers: []
    '''
    with config_context(config):
        reset_logging()
        yield
    reset_logging()
