import collections
import logging
import logging.config as logging_config
import os
from contextlib import contextmanager
import numpy as np
import shutil
from .__version__ import VERSION, VERSION_STATUS, __version__

from mdt.configuration import get_logging_configuration_dict
try:
    logging_config.dictConfig(get_logging_configuration_dict())
except ValueError:
    print('Logging disabled')

from mdt.component_templates.parameters import FreeParameterTemplate, ProtocolParameterTemplate
from mdt.component_templates.cascade_models import CascadeTemplate
from mdt.component_templates.batch_profiles import BatchProfileTemplate
from mdt.component_templates.compartment_models import CompartmentTemplate, WeightCompartmentTemplate
from mdt.component_templates.composite_models import CompositeModelTemplate
from mdt.component_templates.library_functions import LibraryFunctionTemplate

from mdt.lib.model_fitting import get_batch_fitting_function
from mdt.utils import estimate_noise_std, get_cl_devices, load_input_data,\
    create_blank_mask, create_index_matrix, \
    volume_index_to_roi_index, roi_index_to_volume_index, load_brain_mask, init_user_settings, restore_volumes, \
    apply_mask, create_roi, volume_merge, protocol_merge, create_median_otsu_brain_mask, create_brain_mask, \
    load_samples, load_sample, load_nifti, write_slice_roi, apply_mask_to_file, extract_volumes, \
    get_slice_in_dimension, per_model_logging_context, \
    get_temporary_results_dir, get_example_data, SimpleInitializationData, InitializationData, load_volume_maps,\
    covariance_to_correlation, check_user_components, unzip_nifti, zip_nifti
from mdt.lib.sorting import sort_orientations, create_sort_matrix, sort_volumes_per_voxel
from mdt.simulations import create_signal_estimates, simulate_signals, add_rician_noise
from mdt.lib.batch_utils import run_function_on_batch_fit_output, batch_apply, \
    batch_profile_factory, get_subject_selection
from mdt.protocols import load_bvec_bval, load_protocol, auto_load_protocol, write_protocol, write_bvec_bval, \
    create_protocol
from mdt.configuration import config_context, get_processing_strategy, get_config_option, set_config_option
from mdt.lib.exceptions import InsufficientProtocolError
from mdt.lib.nifti import write_nifti
from mdt.lib.components import get_model, get_batch_profile, get_component, get_template


__author__ = 'Robbert Harms'
__date__ = "2015-03-10"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def fit_model(model, input_data, output_folder,
              method=None, recalculate=False, only_recalculate_last=False,
              cl_device_ind=None, double_precision=False, tmp_results_dir=True,
              initialization_data=None, post_processing=None, optimizer_options=None):
    """Run the optimizer on the given model.

    Args:
        model (str or :class:`~mdt.models.composite.DMRICompositeModel` or :class:`~mdt.models.cascade.DMRICascadeModelInterface`):
            An implementation of an AbstractModel that contains the model we want to optimize or the name of
            an model.
        input_data (:class:`~mdt.utils.MRIInputData`): the input data object containing all
            the info needed for the model fitting.
        output_folder (string): The path to the folder where to place the output, we will make a subdir with the
            model name in it.
        method (str): The optimization method to use, one of:
            - 'Levenberg-Marquardt'
            - 'Nelder-Mead'
            - 'Powell'
            - 'Subplex'

            If not given, defaults to 'Powell'.

        recalculate (boolean): If we want to recalculate the results if they are already present.
        only_recalculate_last (boolean):
            This is only of importance when dealing with CascadeModels.
            If set to true we only recalculate the last element in the chain (if recalculate is set to True, that is).
            If set to false, we recalculate everything. This only holds for the first level of the cascade.
        cl_device_ind (int or list): the index of the CL device to use. The index is from the list from the function
            utils.get_cl_devices(). This can also be a list of device indices.
        double_precision (boolean): if we would like to do the calculations in double precision
        tmp_results_dir (str, True or None): The temporary dir for the calculations. Set to a string to use
            that path directly, set to True to use the config value, set to None to disable.
        initialization_data (:class:`~mdt.utils.InitializationData` or dict): provides (extra) initialization data to
            use during model fitting. If we are optimizing a cascade model this data only applies to the last model
            in the cascade. If a dictionary is given we will load the elements as arguments to the
            :class:`mdt.utils.SimpleInitializationData` class. For example::

                initialization_data = {'fixes': {...}, 'inits': {...}}

            is transformed into::

                initialization_data = SimpleInitializationData(fixes={...}, inits={...})
        post_processing (dict): a dictionary with flags for post-processing options to enable or disable.
            For valid elements, please see the configuration file settings for ``optimization``
            under ``post_processing``. Valid input for this parameter is for example: {'covariance': False}
            to disable automatic calculation of the covariance from the Hessian.
        optimizer_options (dict): extra options passed to the optimization routines.

    Returns:
        dict: The result maps for the given composite model or the last model in the cascade.
            This returns the results as 3d/4d volumes for every output map.
    """
    import mdt.utils
    from mdt.lib.model_fitting import ModelFit

    if not mdt.utils.check_user_components():
        init_user_settings(pass_if_exists=True)

    if not isinstance(initialization_data, InitializationData) and initialization_data is not None:
        initialization_data = SimpleInitializationData(**initialization_data)

    if cl_device_ind is not None and not isinstance(cl_device_ind, collections.Iterable):
        cl_device_ind = [cl_device_ind]

    model_fit = ModelFit(model, input_data, output_folder, method=method, optimizer_options=optimizer_options,
                         recalculate=recalculate,
                         only_recalculate_last=only_recalculate_last,
                         cl_device_ind=cl_device_ind, double_precision=double_precision,
                         tmp_results_dir=tmp_results_dir, initialization_data=initialization_data,
                         post_processing=post_processing)
    return model_fit.run()


def sample_model(model, input_data, output_folder, nmr_samples=None, burnin=None, thinning=None,
                 method=None, recalculate=False, cl_device_ind=None, double_precision=False,
                 store_samples=True, sample_items_to_save=None, tmp_results_dir=True,
                 initialization_data=None, post_processing=None, post_sampling_cb=None,
                 sampler_options=None):
    """Sample a composite model using the Adaptive Metropolis-Within-Gibbs (AMWG) MCMC algorithm [1].

    Args:
        model (:class:`~mdt.models.composite.DMRICompositeModel` or str): the model to sample
        input_data (:class:`~mdt.utils.MRIInputData`): the input data object containing all
            the info needed for the model fitting.
        output_folder (string): The path to the folder where to place the output, we will make a subdir with the
            model name in it (for the optimization results) and then a subdir with the samples output.
        nmr_samples (int): the number of samples we would like to return.
        burnin (int): the number of samples to burn-in, that is, to discard before returning the desired
            number of samples
        thinning (int): how many sample we wait before storing a new one. This will draw extra samples such that
                the total number of samples generated is ``nmr_samples * (thinning)`` and the number of samples stored
                is ``nmr_samples``. If set to one or lower we store every sample after the burn in.
        method (str): The sampling method to use, one of:
            - 'AMWG', for the Adaptive Metropolis-Within-Gibbs
            - 'SCAM', for the Single Component Adaptive Metropolis
            - 'FSL', for the sampling method used in the FSL toolbox
            - 'MWG', for the Metropolis-Within-Gibbs (simple random walk metropolis without updates)

            If not given, defaults to 'AMWG'.

        recalculate (boolean): If we want to recalculate the results if they are already present.
        cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
            utils.get_cl_devices().
        double_precision (boolean): if we would like to do the calculations in double precision
        store_samples (boolean): determines if we store any of the samples. If set to False we will store none
            of the samples.
        sample_items_to_save (list): list of output names we want to store the samples of. If given, we only
            store the items specified in this list. Valid items are the free parameter names of the model and the
            items 'LogLikelihood' and 'LogPrior'.
        tmp_results_dir (str, True or None): The temporary dir for the calculations. Set to a string to use
                that path directly, set to True to use the config value, set to None to disable.
        initialization_data (:class:`~mdt.utils.InitializationData` or dict): provides (extra) initialization data to
            use during model fitting. If we are optimizing a cascade model this data only applies to the last model
            in the cascade. If a dictionary is given we will load the elements as arguments to the
            :class:`mdt.utils.SimpleInitializationData` class. For example::

                initialization_data = {'fixes': {...}, 'inits': {...}}

            is transformed into::

                initialization_data = SimpleInitializationData(fixes={...}, inits={...})
        post_processing (dict): a dictionary with flags for post-processing options to enable or disable.
            For valid elements, please see the configuration file settings for ``sample`` under ``post_processing``.
            Valid input for this parameter is for example: {'sample_statistics': True} to enable automatic calculation
            of the sample statistics.
        post_sampling_cb (Callable[
            [mot.sample.base.SamplingOutput, mdt.models.composite.BuildCompositeModel], Optional[Dict]]):
                additional post-processing called after sampling. This function can optionally return a (nested)
                dictionary with as keys dir-/file-names and as values maps to be stored in the results directory.
        sampler_options (dict): specific options for the MCMC routine. These will be provided to the sampling routine
            as additional keyword arguments to the constructor.

    Returns:
        dict: if store_samples is True then we return the samples per parameter as a numpy memmap. If store_samples
            is False we return None

    References:
        1. Roberts GO, Rosenthal JS. Examples of adaptive MCMC. J Comput Graph Stat. 2009;18(2):349-367.
           doi:10.1198/jcgs.2009.06134.
    """
    import mdt.utils
    from mot.lib.load_balance_strategies import EvenDistribution
    from mdt.lib.model_sampling import sample_composite_model
    from mdt.models.cascade import DMRICascadeModelInterface
    import mot.configuration

    settings = mdt.configuration.get_general_sampling_settings()
    if nmr_samples is None:
        nmr_samples = settings['nmr_samples']
    if burnin is None:
        burnin = settings['burnin']
    if thinning is None:
        thinning = settings['thinning']

    if not isinstance(initialization_data, InitializationData) and initialization_data is not None:
        initialization_data = SimpleInitializationData(**initialization_data)

    if not mdt.utils.check_user_components():
        init_user_settings(pass_if_exists=True)

    if isinstance(model, str):
        model = get_model(model)()

    if post_processing:
        model.update_active_post_processing('sampling', post_processing)

    if isinstance(model, DMRICascadeModelInterface):
        raise ValueError('The function \'sample_model()\' does not accept cascade models.')

    if cl_device_ind is not None and not isinstance(cl_device_ind, collections.Iterable):
        cl_device_ind = [cl_device_ind]

    if cl_device_ind is None:
        cl_context_action = mot.configuration.VoidConfigurationAction()
    else:
        cl_envs = [get_cl_devices()[ind] for ind in cl_device_ind]
        cl_context_action = mot.configuration.RuntimeConfigurationAction(
            cl_environments=cl_envs,
            load_balancer=EvenDistribution(),
            double_precision=double_precision)

    with mot.configuration.config_context(cl_context_action):
        base_dir = os.path.join(output_folder, model.name, 'samples')

        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)

        if recalculate:
            shutil.rmtree(base_dir)

        logger = logging.getLogger(__name__)
        logger.info('Using MDT version {}'.format(__version__))
        logger.info('Preparing for model {0}'.format(model.name))
        logger.info('The {0} parameters we will sample are: {1}'.format(len(model.get_free_param_names()),
                                                                            model.get_free_param_names()))

        return sample_composite_model(model, input_data, base_dir, nmr_samples, thinning, burnin,
                                      get_temporary_results_dir(tmp_results_dir),
                                      method=method, recalculate=recalculate,
                                      store_samples=store_samples,
                                      sample_items_to_save=sample_items_to_save,
                                      initialization_data=initialization_data,
                                      post_sampling_cb=post_sampling_cb,
                                      sampler_options=sampler_options)


def batch_fit(data_folder, models_to_fit, output_folder=None, batch_profile=None,
              subjects_selection=None, recalculate=False,
              cl_device_ind=None, dry_run=False,
              double_precision=False, tmp_results_dir=True,
              use_gradient_deviations=False):
    """Run all the available and applicable models on the data in the given folder.

    The idea is that a single folder is enough to fit_model the computations. One can optionally give it the
    batch_profile to use for the fitting. If not given, this class will attempt to use the
    batch_profile that fits the data folder best.

    Args:
        data_folder (str): The data folder to process
        models_to_fit (list of str): A list of models to fit to the data.
        output_folder (str): the folder in which to place the output, if not given we per default put an output folder
            next to the data_folder.
        batch_profile (:class:`~mdt.lib.batch_utils.BatchProfile` or str): the batch profile to use,
            or the name of a batch profile to use. If not given it is auto detected.
        subjects_selection (:class:`~mdt.lib.batch_utils.BatchSubjectSelection` or iterable): the subjects to \
            use for processing. If None, all subjects are processed. If a list is given instead of a
            :class:`~mdt.lib.batch_utils.BatchSubjectSelection` instance, we apply the following. If the elements in that
            list are string we use it as subject ids, if they are integers we use it as subject indices.
        recalculate (boolean): If we want to recalculate the results if they are already present.
        cl_device_ind (int or list of int): the index of the CL device to use.
            The index is from the list from the function get_cl_devices().
        dry_run (boolean): a dry run will do no computations, but will list all the subjects found in the
            given directory.
        double_precision (boolean): if we would like to do the calculations in double precision
        tmp_results_dir (str, True or None): The temporary dir for the calculations. Set to a string to use
                that path directly, set to True to use the config value, set to None to disable.
        use_gradient_deviations (boolean): if you want to use the gradient deviations if present
    Returns:
        The list of subjects we will calculate / have calculated.
    """
    logger = logging.getLogger(__name__)

    if not check_user_components():
        init_user_settings(pass_if_exists=True)

    if output_folder is None:
        output_folder = os.path.join(data_folder + '/', '..', os.path.dirname(data_folder + '/') + '_output')

    batch_profile = batch_profile_factory(batch_profile, data_folder)
    if batch_profile is None:
        raise RuntimeError('No suitable batch profile could be '
                           'found for the directory {0}'.format(os.path.abspath(data_folder)))
    subjects_selection = get_subject_selection(subjects_selection)

    logger.info('Using MDT version {}'.format(__version__))
    logger.info('Using batch profile: {0}'.format(batch_profile))

    if dry_run:
        logger.info('Dry run enabled')

    all_subjects = batch_profile.get_subjects(data_folder)
    subjects = subjects_selection.get_subjects(batch_profile.get_subjects(data_folder))
    logger.info('Fitting models: {}'.format(models_to_fit))
    logger.info('Subjects found: {0}'.format(len(all_subjects)))
    logger.info('Subjects to process: {0}'.format(len(subjects)))

    if dry_run:
        logger.info('Subjects found: {0}'.format(list(subject.subject_id for subject in subjects)))
        return

    batch_fit_func = get_batch_fitting_function(
        len(subjects), models_to_fit, output_folder, recalculate=recalculate,
        cl_device_ind=cl_device_ind, double_precision=double_precision,
        tmp_results_dir=tmp_results_dir, use_gradient_deviations=use_gradient_deviations)

    return batch_apply(batch_fit_func, data_folder, batch_profile=batch_profile, subjects_selection=subjects_selection)


def view_maps(data, config=None, figure_options=None,
              block=True, show_maximized=False, use_qt=True,
              window_title=None, save_filename=None):
    """View a number of maps using the MDT Maps Visualizer.

    Args:
        data (str, dict, :class:`~mdt.visualization.maps.base.DataInfo`, list, tuple): the data we are showing,
            either a dictionary with result maps, a string with a path name, a DataInfo object or a list
            with filenames and/or directories.
        config (str, dict, :class:`~mdt.visualization.maps.base import MapPlotConfig`): either a Yaml string or a
            dictionary with configuration settings or a ValidatedMapPlotConfig object to use directly
        figure_options (dict): Used when ``use_qt`` is False or when ``write_figure`` is used.
            Sets the figure options for the matplotlib Figure.
            If figsizes is not given you can also specify two ints, width and height, to indicate the pixel size of
            the resulting figure, together with the dpi they are used to calculate the figsize.
        block (boolean): if we block the plots or not
        show_maximized (boolean): if we show the window maximized or not
        window_title (str): the title for the window
        use_qt (boolean): if we want to use the Qt GUI, or show the results directly in matplotlib
        save_filename (str): save the figure to file. If set, we will not display the viewer.
    """
    from mdt.gui.maps_visualizer.main import start_gui
    from mdt.visualization.maps.base import MapPlotConfig
    from mdt.visualization.maps.matplotlib_renderer import MapsVisualizer
    from mdt.visualization.maps.base import SimpleDataInfo
    import matplotlib.pyplot as plt
    from mdt.gui.maps_visualizer.actions import NewDataAction
    from mdt.gui.maps_visualizer.base import SimpleDataConfigModel

    if isinstance(data, str):
        data = SimpleDataInfo.from_paths([data])
    elif isinstance(data, collections.MutableMapping):
        data = SimpleDataInfo(data)
    elif isinstance(data, collections.Sequence):
        if all(isinstance(el, str) for el in data):
            data = SimpleDataInfo.from_paths(data)
        else:
            data = SimpleDataInfo({str(ind): v for ind, v in enumerate(data)})
    elif data is None:
        data = SimpleDataInfo({})

    if config is None:
        config = MapPlotConfig()
    elif isinstance(config, str):
        if config.strip():
            config = MapPlotConfig.from_yaml(config)
        else:
            config = MapPlotConfig()
    elif isinstance(config, dict):
        config = MapPlotConfig.from_dict(config)

    config = config.create_valid(data)

    if not use_qt or save_filename:
        settings = {'dpi': 100, 'width': 1800, 'height': 1600}
        if save_filename:
            settings = {'dpi': 80, 'width': 800, 'height': 600}

        settings.update(figure_options or {})
        if 'figsize' not in settings:
            settings['figsize'] = (settings['width'] / settings['dpi'],
                                   settings['height'] / settings['dpi'])

        del settings['width']
        del settings['height']

        figure = plt.figure(**settings)
        viz = MapsVisualizer(data, figure)

        if save_filename:
            viz.to_file(save_filename, config, dpi=settings['dpi'])
        else:
            viz.show(config, block=block, maximize=show_maximized)
    else:
        start_gui(data, config, app_exec=block, show_maximized=show_maximized, window_title=window_title)


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
        kwargs (kwargs): see SampleVisualizer for all the supported keywords
    """
    from mdt.visualization.samples import SampleVisualizer

    if isinstance(data, str):
        data = load_samples(data)

    if not data:
        raise ValueError('No samples provided.')

    if kwargs.get('voxel_ind') is None:
        kwargs.update({'voxel_ind': data[list(data.keys())[0]].shape[0] / 2})
    SampleVisualizer(data).show(**kwargs)


def make_path_joiner(*args, make_dirs=False):
    """Generates and returns an instance of utils.PathJoiner to quickly join path names.

    Args:
        *args: the initial directory or list of directories to concatenate
        make_dirs (boolean): if we should make the referenced directory if it does not yet exist

    Returns:
         mdt.utils.PathJoiner: easy path manipulation path joiner
    """
    from mdt.utils import PathJoiner
    return PathJoiner(*args, make_dirs=make_dirs)


def sort_maps(input_maps, reversed_sort=False, sort_index_matrix=None):
    """Sort the values of the given maps voxel by voxel.

    This first creates a sort matrix to index the maps in sorted order per voxel. Next, it creates the output
    maps for the maps we sort on.

    Args:
        input_maps (:class:`list`): a list of string (filenames) or ndarrays we will sort
        reversed_sort (boolean): if we want to sort from large to small instead of small to large.
            This is not used if a sort index matrix is provided.
        sort_index_matrix (ndarray): if given we use this sort index map instead of generating one by sorting the
            maps_to_sort_on. Supposed to be a integer matrix.

    Returns:
        list: the list of sorted volumes
    """
    if sort_index_matrix is None:
        sort_index_matrix = create_sort_matrix(input_maps, reversed_sort=reversed_sort)
    elif isinstance(sort_index_matrix, str):
        sort_index_matrix = np.round(load_nifti(sort_index_matrix).get_data()).astype(np.int64)
    return sort_volumes_per_voxel(input_maps, sort_index_matrix)


def get_volume_names(directory):
    """Get the names of the Nifti volume maps in the given directory.

    Args:
        directory: the directory to get the names of the available maps from.

    Returns:
        :class:`list`: A list with the names of the volumes.
    """
    from mdt.lib.nifti import yield_nifti_info
    return list(sorted(el[1] for el in yield_nifti_info(directory)))


def write_volume_maps(maps, directory, header=None, overwrite_volumes=True, gzip=True):
    """Write a dictionary with maps to the given directory using the given header.

    Args:
        maps (dict): The maps with as keys the map names and as values 3d or 4d maps
        directory (str): The dir to write to
        header: The Nibabel Image Header
        overwrite_volumes (boolean): If we want to overwrite the volumes if they are present.
        gzip (boolean): if we want to write the results gzipped
    """
    from mdt.lib.nifti import write_all_as_nifti
    write_all_as_nifti(maps, directory, nifti_header=header, overwrite_volumes=overwrite_volumes, gzip=gzip)


def get_models_list():
    """Get a list of all available models, composite and cascade.

    Returns:
        list of str: A list of available model names.
    """
    from mdt.lib.components import list_composite_models, list_cascade_models
    l = list(list_cascade_models())
    l.extend(list_composite_models())
    return list(sorted(l))


def get_models_meta_info():
    """Get the meta information tags for all the models returned by get_models_list()

    Returns:
        dict of dict: The first dictionary indexes the model names to the meta tags, the second holds the meta
            information.
    """
    from mdt.lib.components import list_cascade_models, list_composite_models, get_meta_info, get_component_list
    meta_info = {}
    for model_type in ('composite_models', 'cascade_models'):
        model_list = get_component_list(model_type)
        for model in model_list:
            meta_info.update({model: get_meta_info(model_type, model)})
    return meta_info


def start_gui(base_dir=None, app_exec=True):
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
def with_logging_to_debug():
    """A context in which the logging is temporarily set to WARNING.

    Example of usage::

        with mdt.with_logging_to_debug():
            your_computations()

    During the function ``your_computations`` only WARNING level logging will show up.
    """
    handlers = logging.getLogger('mot').handlers
    previous_levels = [handler.level for handler in handlers]
    for handler in handlers:
        handler.setLevel(logging.WARNING)
    yield
    for handler, previous_level in zip(handlers, previous_levels):
        handler.setLevel(previous_level)


def reload_components():
    """Reload all the dynamic components.

    This can be useful after changing some of the dynamically loadable modules. This function will remove all cached
    components and reload the directories.
    """
    from mdt.lib.components import reload
    try:
        reload()
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.error('Failed to load the default components. Try removing your MDT home folder and reload.')


if 'MDT.LOAD_COMPONENTS' in os.environ and os.environ['MDT.LOAD_COMPONENTS'] != '1':
    pass
else:
    reload_components()
