import os
import pickle
import re
import matplotlib.pyplot as plt
from six import string_types
import mdt.batch_utils
from mdt.cascade_model import CascadeModelInterface
import mdt.masking as masking
from mdt.model_fitting import ModelFit, BatchFitting
from mdt.components_loader import SingleModelsLoader, CascadeModelsLoader
import mdt.configuration
from mdt import utils
from mdt.IO import Nifti, TrackMark
import nibabel as nib
from mdt.model_sampling import sample_single_model
from mdt.protocols import load_from_protocol, load_bvec_bval
import mdt.protocols as protocols
from mdt.utils import concatenate_two_mri_measurements, DMRIProblemData, setup_logging, configure_per_model_logging
import mdt.utils
from mdt.visualization import SampleVisualizer, MapsVisualizer
import numpy as np
from mot import runtime_configuration
from mot.cl_routines.sampling.metropolis_hastings import MetropolisHastings

__author__ = 'Robbert Harms'
__date__ = "2015-03-10"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


VERSION = '0.2.1'
VERSION_STATUS = ''

_items = VERSION.split('-')                                           
VERSION_NUMBER_PARTS = tuple(int(i) for i in _items[0].split('.'))
if len(_items) > 1:
    VERSION_STATUS = _items[1]
__version__ = VERSION


try:
    setup_logging()
except ValueError:
    print('Logging disabled')


def batch_fit(data_folder, batch_profile_class=None, subjects_ind=None, recalculate=False,
              cl_device_ind=None, dry_run=False):
    """Run all the available and applicable models on the data in the given folder.

    See the class AutoRun for more details and options.

    Setting the cl_device_ind has the side effect that it changes the current run time cl_device settings in the MOT
    toolkit.

    Args:
        data_folder (str): The data folder to process
        batch_profile_class (BatchProfile class or str): the batch profile class to use, can also be the name
            of a batch profile to load. If not given it is auto detected.
            Please note it expects a callable that returns a batch profile instance. For example, you can use it as:
                batch_profile_class=MyBatchProfile
            but this would not work:
                batch_profile_class=MyBatchProfile()
        subjects_ind (list of int): either a list of subjects to process or the index of a single subject to process.
            To get a list of subjects run this function with the dry_run parameter to true.
        recalculate (boolean): If we want to recalculate the results if they are already present.
        cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
            get_cl_devices().
        dry_run (boolean): a dry run will do no computations, but will list all the subjects found in the
            given directory.

    Returns:
        The list of subjects we will calculate / have calculated.
    """
    if not utils.check_user_components():
        raise RuntimeError('User\'s components folder is not up to date. Please run the script mdt-init-user-settings.')

    batch_fitting = BatchFitting(data_folder, batch_profile_class=batch_profile_class, subjects_ind=subjects_ind,
                                 recalculate=recalculate, cl_device_ind=cl_device_ind)

    if dry_run:
        return batch_fitting.get_subjects_info()

    return batch_fitting.run()


def fit_model(model, dwi_info, protocol, brain_mask, output_folder, optimizer=None,
              recalculate=False, only_recalculate_last=False, cl_device_ind=None):
    """Run the optimizer on the given model.

    Args:
        model (str or AbstractModel): An implementation of an AbstractModel that contains the model we want to optimize
            or the name of an model we load with get_model()
        dwi_info (string): Either an (ndarray, img_header) tuple or the full path to the volume (4d signal data).
        protocol (Protocol or string): A protocol object with the right protocol for the given data,
            or a string object with a filename to the given file.
        brain_mask (string): A full path to a mask file that can optionally be used. If None given, no mask is used.
        output_folder (string): The path to the folder where to place the output, we will make a subdir with the
            model name in it.
        optimizer (AbstractOptimizer): The optimization routine to use.
        recalculate (boolean): If we want to recalculate the results if they are already present.
        only_recalculate_last (boolean):
            This is only of importance when dealing with CascadeModels.
            If set to true we only recalculate the last element in the chain (if recalculate is set to True, that is).
            If set to false, we recalculate everything. This only holds for the first level of the cascade.
        cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
            utils.get_cl_devices().

    Returns:
        the output of the optimization. If a cascade is given, only the results of the last model in the cascade is
        returned
    """
    if not utils.check_user_components():
        raise RuntimeError('User\'s components folder is not up to date. Please the script mdt-init-user-settings.')

    model_fit = ModelFit.load_from_basic_data(
        model, dwi_info, protocol, brain_mask, output_folder, optimizer=optimizer,
        recalculate=recalculate, only_recalculate_last=only_recalculate_last, cl_device_ind=cl_device_ind)
    return model_fit.run()


def sample_model(model, dwi_info, protocol, brain_mask, output_folder,
                 sampler=None, recalculate=False, cl_device_ind=None):
    """Sample a single model. This does not accept cascade models, only single models.

    Args:
        model: the model to sample
        dwi_info (string): Either an (ndarray, img_header) tuple or the full path to the volume (4d signal data).
        protocol (Protocol or string): A protocol object with the right protocol for the given data,
            or a string object with a filename to the given file.
        brain_mask (string): A full path to a mask file that can optionally be used. If None given, no mask is used.
        output_folder (string): The path to the folder where to place the output, we will make a subdir with the
            model name in it (for the optimization results) and then a subdir with the samples output.
        sampler (AbstractSampler): the sampler to use
        recalculate (boolean): If we want to recalculate the results if they are already present.
        cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
            utils.get_cl_devices().

    Returns:
        the full chain of the optimization
    """
    if not utils.check_user_components():
        raise RuntimeError('User\'s components folder is not up to date. Please run mdt.initialize_user_settings().')

    if isinstance(model, CascadeModelInterface):
        raise ValueError('The function \'sample_model()\' does not accept cascade models.')

    if cl_device_ind is not None:
        runtime_configuration.runtime_config['cl_environments'] = [get_cl_devices()[cl_device_ind]]

    problem_data = load_problem_data(dwi_info, protocol, brain_mask)
    if sampler is None:
        sampler = MetropolisHastings(runtime_configuration.runtime_config['cl_environments'],
                                     runtime_configuration.runtime_config['load_balancer'])

    return sample_single_model(model, problem_data, output_folder, sampler, recalculate=recalculate)


def update_config(config):
    """Update the runtime configuration in mdt.configuration with the given config values.

    Args:
        config (dict): a dictionary with configuration options that will overwrite the current configuration.
    """
    configuration.load_from_dict(config)


def get_config():
    """Get the the current runtime configuration

    Returns:
        dict: the current run time configuration.
    """
    return configuration.config


def collect_batch_fit_output(data_folder, output_dir, batch_profile_class=None, mask_name=None, symlink=False):
    """Load from the given data folder all the output files and put them into the output directory.

    If there is more than one mask file available the user has to choose which mask to use using the mask_name
    keyword argument. If it is not given an error is raised.

    The results for the chosen mask it placed in the output folder per subject. Example:
        <output_dir>/<subject_id>/<results>

    Args:
        data_folder (str): The data folder with the output files
        output_dir (str): The path to the output folder where all the files will be put.
        batch_profile_class (BatchProfile class or str): the batch profile class to use, can also be the name
            of a batch profile to load. If not given it is auto detected.
            Please note it expects a callable that returns a batch profile instance. For example, you can use it as:
                batch_profile_class=MyBatchProfile
            but this would not work:
                batch_profile_class=MyBatchProfile()
        mask_name (str): the mask to use to get the output from
        symlink (boolean): only available under Unix OS's. Creates a symlink instead of copying.
    """
    mdt.batch_utils.collect_batch_fit_output(data_folder, output_dir, batch_profile_class=batch_profile_class,
                                   mask_name=mask_name, symlink=symlink)


def run_function_on_batch_fit_output(data_folder, func, batch_profile_class=None):
    """Run a function on the output of a batch fitting routine.

    This enables you to run a function on every model output from every subject. The python function should accept
    as single argument an instance of the class BatchFitSubjectOutputInfo.

    Args:
        data_folder (str): The data folder with the output files
        func (python function): the python function we should call for every map and model
        batch_profile_class (BatchProfile class or str): the batch profile class to use, can also be the name
            of a batch profile to load. If not given it is auto detected.
            Please note it expects a callable that returns a batch profile instance. For example, you can use it as:
                batch_profile_class=MyBatchProfile
            but this would not work:
                batch_profile_class=MyBatchProfile()
    """
    mdt.batch_utils.run_function_on_batch_fit_output(data_folder, func, batch_profile_class=batch_profile_class)


def get_cl_devices():
    """Get a list of all CL devices in the system.

    The indices of the devices can be used in the model fitting functions for 'cl_device_ind'.

    Returns:
        A list of CLEnvironments, one for each device in the system.
    """
    return mdt.utils.get_cl_devices()


def load_problem_data(volume_info, protocol, mask):
    """Load and create the problem data object that can be given to a model

    Args:
        volume_info (string): Either an (ndarray, img_header) tuple or the full path to the volume (4d signal data).
        protocol (Protocol or string): A protocol object with the right protocol for the given data,
            or a string object with a filename to the given file.
        mask (string): A full path to a mask file that can optionally be used. If None given, no mask is used.

    Returns:
        The Problem data, in the ProblemData container object.
    """
    return utils.load_problem_data(volume_info, protocol, mask)


def load_protocol_bval_bvec(bvec=None, bval=None, bval_scale='auto'):
    """Create a protocol out of a bvac filename and a bvec filename.

    Args:
        bval (string): The bval filename
        bvec (string): The bvec filename
        bval_scale (double): The scale by which to scale the values in the bval file.

    Returns:
        A Protocol object.
    """
    return load_bvec_bval(bvec, bval, bval_scale=bval_scale)


def load_protocol(filename, column_names=None):
    """Load an protocol from the given protocol file, with as column names the given list of names.

    Args:
        protocol_fname (string):
            The filename of the protocol file to load. This should be a comma seperated, or tab delimited file
            with equal length columns. The column names can go on the tab and should be comma or space seperated.
        column_names (tuple, optional):
            A tuple or list of the columns names. Please note that every column should be named. The gradient vector
            for example should be listed as 'gx', 'gy', 'gz'.

    Returns:
        An protocol with all the columns loaded.
    """
    return load_from_protocol(filename, column_names)


def write_protocol(protocol, fname, columns_list=None):
    """Write a protocol to a file.

    Args:
        protocol (Protocol): The protocol object information to write
        fname (string): The filename to write to
        columns_list (list, optional): Only write these columns (and in this order).
    """
    protocols.write_protocol(protocol, fname, columns_list=columns_list)


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
    protocols.write_bvec_bval(protocol, bvec_fname, bval_fname, column_based=column_based, bval_scale=bval_scale)


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

    Returns:
        ndarray: The created brain mask
    """
    if output_fname:
        if not isinstance(dwi_info, (string_types, tuple, list)):
            raise ValueError('No header obtainable, can not write the brain mask.')
        return masking.create_write_median_otsu_brain_mask(dwi_info, protocol, output_fname, **kwargs)

    return masking.create_median_otsu_brain_mask(dwi_info, protocol, **kwargs)


def load_brain_mask(brain_mask_fname):
    """Load the brain mask from the given file.

    Args:
        brain_mask_fname (string): The filename of the brain mask to load.

    Returns:
        The loaded brain mask data
    """
    return utils.load_brain_mask(brain_mask_fname)


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

        protocol = load_from_protocol(e['protocol'])
        to_concat.append((protocol, signal4d))

    protocol, signal4d = concatenate_two_mri_measurements(to_concat)
    nib.Nifti1Image(signal4d, None, nii_header).to_filename(output_volume_fname)
    protocols.write_protocol(protocol, output_protocol_fname)


def view_results_slice(data,
                       dimension=None,
                       slice_ind=None,
                       maps_to_show=None,
                       general_plot_options=None,
                       map_plot_options=None,
                       font_size=None,
                       to_file=None,
                       block=True,
                       maximize=False,
                       window_title=None):
    """View from the given results the given slice.

    Args:
        data (string or dict with maps): Either the folder to read the maps from, or the data itself in a dictionary.
        dimension (int): the dimension to select a slice from
        slice (int): the selected slice
        maps_to_show (list): A list of parameters we want to show (in that order)
        general_plot_options (dict): A number of options for rendering the maps. These hold for all the displayed maps.
        map_plot_options (dict): A number of options for rendering the maps. These options should be like:
                {map_name: {options}}.
                That is a set of options for that specific map. These override the general plot options if present.
                For example, to change the contrast one can use as options: {'vmin': 0, 'vmax': 1}
        font_size (int): The size of the font for larger or smaller print.
        to_file (string, optional, default None): If to_file is not None it is supposed to be a filename where the
            images will be saved and nothing will be displayed. If set to None, no file is saved and results are
            displayed. Already existing items will be overwritten.
        block (boolean): If we want to block after calling the plots or not. Set this to False if you
            do not want the routine to block after drawing. In doing so you manually need to block. You can
            do this by calling the function block_plots()
        maximize (boolean): if we want to display the window maximized or not
        window_title (str): the title of the window. If None, the default title is used
    """
    general_plot_options = general_plot_options or {}

    if 'cmap' not in general_plot_options:
        general_plot_options.update({'cmap': 'hot'})

    if isinstance(data, string_types):
        results_total = load_volume_maps(data)
    else:
        results_total = data

    if maps_to_show:
        results_total = {k: results_total[k] for k in maps_to_show}

    viz = MapsVisualizer(results_total)
    if font_size:
        viz.font_size = font_size
    viz.show(dimension=dimension,
             slice_ind=slice_ind,
             maps_to_show=maps_to_show,
             general_plot_options=general_plot_options,
             map_plot_options=map_plot_options,
             to_file=to_file,
             block=block,
             maximize=maximize,
             window_title=window_title)


def block_plots():
    """A small function to block the plots made by matplotlib.

    This basically only calls plt.show()
    """
    plt.show()


def view_result_samples(data, voxel_ind=None, block=True):
    """View the samples from the given results set.

    Args:
        data (string or dict): The location of the maps to load the samples from, or the samples themselves.
        voxel_ind (int): The index of the voxel to choose. The samples are still in ROI space,
            so a linear index suffices.
        block (boolean): If we want to block after calling the plots or not. Set this to False if you
            do not want the routine to block after drawing. In doing so you manually need to block. You can
            do this by calling the function block_plots()
    """
    if isinstance(data, string_types):
        with open(data, 'rb') as f:
            data = pickle.load(f)

    if voxel_ind is None:
        voxel_ind = data[data.keys()[0]].shape[0] / 2
    SampleVisualizer(data).show(voxel_ind=voxel_ind, block=block)


def load_dwi(volume_fname):
    """Load the diffusion weighted image data from the given volume filename.

    This does not perform any data type changes, so the input may not be in float64. If you call this function
    to satisfy load_problem_data() then this is not a problem.

    Args:
        volume_fname (string): The filename of the volume to load.

    Returns:
        a tuple with (data, header) for the given file.
    """
    return utils.load_dwi(volume_fname)


def load_nifti(nifti_volume):
    """Load and return a nifti file.

    A more general function than load_dwi which is meant for raw diffusion images.
    Args:
        volume_fname (string): The filename of the volume to load.

    Returns:
        nib image
    """
    return nib.load(nifti_volume)


def make_path_joiner(*folder):
    """Generates and returns an instance of utils.PathJoiner to quickly join pathnames.

    Returns:
        An instance of utils.PathJoiner for easy path manipulation.
    """
    return utils.PathJoiner(*folder)


def create_roi(data, brain_mask):
    """Create and return the region of interest of the given brain volume and mask

    Args:
        data: a brain volume with four dimensions (x, y, z, w) where w is the length of the protocol, or a list
                tuple or dictionary with volumes
        brain_mask: the mask indicating the region of interest, dimensions: (x, y, z)

    Returns:
        Signal lists for each of the given volumes. The axis are: (voxels, protocol)
    """
    return utils.create_roi(data, brain_mask)


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
    if os.path.exists(output_fname) and not overwrite_if_exists:
        return load_brain_mask(output_fname)
    brain_mask_img = nib.load(brain_mask_fname)
    brain_mask = brain_mask_img.get_data()
    img_header = brain_mask_img.get_header()
    roi_mask = utils.create_slice_roi(brain_mask, roi_dimension, roi_slice)
    write_image(output_fname, roi_mask, img_header)
    return roi_mask


def write_image(fname, data, header):
    """Write the given data with the given header to the given file.

    Args:
        fname (str): The output filename
        data (ndarray): The data to write
        header (nibabel header): The header to use
    """
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
    return utils.restore_volumes(data, brain_mask, with_volume_dim=with_volume_dim)


def write_trackmark_files(input_folder, output_folder=None, eigenvalue_scalar=1e6, tvl_header=(1, 1.8, 0, 0),
                          eigen_pairs=None):
    """Convert the given output directory to TrackMark files (a proprietary software package from Alard Roebroeck).

    Basically only the input directory is necessary and it will write the TVL and Rawmaps to a subdirectory.

    The eigenvalues and vectors are searched by searching for pairs of files like:
        <model_name>.eig<ind>.vec.nii(.gz)
        <model_name>.eig<ind>.val.nii(.gz)

    For example:
        Tensor.eig0.vec.nii.gz (map with 3 directions for each voxel)
        Tensor.eig0.val.nii.gz (map with 1 value for each voxel)

    It will search until it has up to three different eigen pairs. It will then order them by name and use them in
    sorted by name order.

    To specifically tell this function which eigenpairs to use you can use the parameter eigen_pairs to give a list of
    eigen pairs that must be used in that order. The list should specify filenames of the maps to use.

    It is even possible to let this function write maps for eigenvectors of which each component of the vector has it's
    own map file. Again, use the correct notation of the parameter eigen_pairs for this.

    Next to writing the TVL file this will also write rawmaps for every map in the directory,

    If there is no sufficient material for creating a TVL, only rawmaps are created.

    Args:
        input_folder (str): The location of the input folder with all the Nifti maps.
        output_folder (str): The output folder. If not given set to a subfolder 'trackmark' in the input folder.
        eigenvalue_scalar (double): The scalar by which we want to scale all the eigenvalues. Trackmark accepts the
            eigenvalues in units of mm^2, while we work in m^2. So we scale our results by 1e6 in general.
        tvl_header (list): The list with header arguments for writing the TVL. See IO.TrackMark for specifics.
        eigen_pairs (list): The optional list with specific eigenvalues and vectors (all volume key names)
            to use for the TVL.
            This can either be a list like: ((vec, val), (vec1, val1), ...) or instead of a single vector file it can
            also be like (vec0, vec1, vec2). For example: (((vec0, vec1, vec2), val), (vec1, val1), ...)

    Returns:
        None
    """
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'trackmark')

    volumes = load_volume_maps(input_folder)

    if eigen_pairs is None:
        eigen_vectors_keys = sorted([k for k in volumes.keys() if re.match(r'(.*)\.eig[0-9]*\.vec$', k)])
        eigen_values_keys = sorted([k for k in volumes.keys() if re.match(r'(.*)\.eig[0-9]*\.val$', k)])

        eigen_vectors = [volumes[k] for k in eigen_vectors_keys]
        eigen_values = [volumes[k] for k in eigen_values_keys]

        eigen_pairs = zip(eigen_vectors, eigen_values)
    else:
        eigen_pairs_keys = eigen_pairs
        eigen_pairs = []
        for eigen_pair in eigen_pairs_keys:
            if isinstance(eigen_pair[0], (list, tuple)):
                vec = np.concatenate([np.expand_dims(volumes[k], axis=3) for k in eigen_pair[0]], axis=3)
            else:
                vec = volumes[eigen_pair[0]]

            val = volumes[eigen_pair[1]]
            eigen_pairs.append((vec, val))

    if eigenvalue_scalar is not None:
        for eigen_pair in eigen_pairs:
            vals = eigen_pair[1]
            vals *= eigenvalue_scalar

    if eigen_pairs:
        TrackMark.write_tvl_direction_pairs(os.path.join(output_folder, 'master.tvl'), tvl_header, eigen_pairs)
    TrackMark.write_rawmaps(output_folder, volumes)


def load_volume_maps(directory, map_names=None):
    """Read a number of Nifti volume maps that were written using write_volume_maps.

    Args:
        directory: the directory from which we want to read a number of maps
        map_names: the names of the maps we want to load. If given we only load and return these maps.

    Returns:
        dict: A dictionary with the volumes. The keys of the dictionary are the filenames (without the extension) of the
            files in the given directory.
    """
    return Nifti.read_volume_maps(directory, map_names=map_names)


def get_volume_names(directory):
    """Get the names of the Nifti volume maps in the given directory.

    Args:
        directory: the directory to get the names of the available maps from.

    Returns:
        list: A list with the names of the volumes.
    """
    return list(sorted(Nifti.volume_names_generator(directory)))


def write_volume_maps(maps, directory, header, overwrite_volumes=True):
    """Write a dictionary with maps to the given directory using the given header.

    Args:
        maps (dict): The maps with as keys the map names and as values 3d or 4d maps
        directory (str): The dir to write to
        header: The Nibabel Image Header
        overwrite_volumes (boolean): If we want to overwrite the volumes if they are present.
    """
    return Nifti.write_volume_maps(maps, directory, header, overwrite_volumes=overwrite_volumes)


def get_list_of_single_models():
    """Get a list of all available single models

    Returns:
        list of str: A list of all available single model names.
    """
    return SingleModelsLoader().list_all()


def get_list_of_cascade_models():
    """Get a list of all available cascade models

    Returns:
        list of str: A list of available cascade models
    """
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
    sml = SingleModelsLoader()
    cml = CascadeModelsLoader()

    meta_info = {}
    for x in (sml, cml):
        models = x.list_all()
        for model in models:
            meta_info.update({model: x.get_meta_info(model)})
    return meta_info


def get_model(model_name, **kwargs):
    """Load one of the available models.

    Args:
        model_name (str): One of the models from get_list_of_single_models() or get_list_of_cascade_models()
        **kwargs: Extra keyword arguments used for the initialization of the model

    Returns:
        Either a cascade model or a single model. In any case, a model that can be given to the fit_model function.
    """
    return components_loader.get_model(model_name, **kwargs)


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
        volumes.update({basename + '_split_' + repr(split_dimension) + '_' + lengths[ind]: v})

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
    if isinstance(input_protocol, basestring):
        input_protocol = mdt.load_protocol(input_protocol)

    new_protocol = input_protocol.get_new_protocol_with_indices(protocol_indices)
    protocols.write_protocol(new_protocol, output_protocol)

    input_volume = load_dwi(input_volume_fname)
    image_data = input_volume[0][..., protocol_indices]
    write_image(output_volume_fname, image_data, input_volume[1])


def extract_tensor_shell(input_volume_fname, input_protocol, output_volume_fname, output_protocol, max_b_val=1.0e9):
    """ Extract the tensor shell from the given volume and protocol

    Args:
        input_volume_fname (str): the input volume from which to get the specific volumes
        input_protocol (str): the input protocol filename
        output_volume_fname (str): the output filename for the selected volumes
        output_protocol (str): the output protocol for the selected volumes
        max_b_val (double): the maximum b-value, standard is 1.0e-9
    """
    protocol = mdt.load_protocol(input_protocol)
    tensor_shell_ind = protocol.get_indices_bval_in_range(start=10, end=max_b_val)
    tensor_shell_ind = np.unique(np.append(tensor_shell_ind, protocol.get_unweighted_indices()))
    mdt.extract_volumes(input_volume_fname, protocol, output_volume_fname, output_protocol, tensor_shell_ind)


def apply_mask(dwi, mask, inplace=False):
    """Apply a mask to the given input.

    Args:
        input_fname (str or ndarray): The input file path or the image itself
        mask_fname (str or ndarray): The filename of the mask or the mask itself
        inplace (boolean): if true we apply the mask in place on the dwi image. If false we do not.
            The default is False.

    Returns:
        ndarray: image of the same size as the input image but with all values set to zero where the mask is zero.
    """
    if isinstance(mask, string_types):
        mask = load_brain_mask(mask)

    if isinstance(dwi, string_types):
        dwi = load_dwi(dwi)[0]

    mask = mask.reshape(mask.shape + (dwi.ndim - mask.ndim) * (1,))

    if inplace:
        dwi *= mask
        return dwi
    return dwi * mask


def apply_mask_to_file(input_fname, mask_fname, output_fname=None):
    """Apply a mask to the given input (nifti) file.

    If no output filename is given, the input file is overwritten.

    Args:
        input_fname (str): The input file path
        mask_fname (str): The filename of the mask
        output_fname (str): The filename for the output file (the masked input file).
    """
    if output_fname is None:
        output_fname = input_fname

    mask = load_brain_mask(mask_fname)
    image_info = load_dwi(input_fname)
    mask = mask.reshape(mask.shape + (image_info[0].ndim - mask.ndim) * (1,))
    masked = image_info[0]
    masked *= mask
    write_image(output_fname, masked, image_info[1])


def initialize_user_settings(overwrite=True):
    """Initializes the user settings folder using a skeleton.

    This will create all the necessary directories for adding components to MDT. It will also create a basic
    configuration file for setting global wide MDT options.

    If the users home folder already exists a backup copy is created first,

    Args:
        overwrite (boolean): If we want to overwrite the folder if it already exists. If true we overwrite, if false
            we do not.

    Returns:
        the path the user settings skeleton was written to
    """
    return utils.initialize_user_settings(overwrite)

def dwi_merge(dwi_images, output_fname, sort=True):
    """ Merge a list of DWI images on the 4th dimension. Writes the result as a file.

    Please note that by default this will sort the list of DWI names based on a natural key sort. This is
    the most convenient option in the case of globbing files. You can disable this behaviour by setting the keyword
    argument 'sort' to False

    Example usage is:
        mdt.dwi_merge(glob.glob(pjoin('raw', '*b278*.nii')),
                      pjoin('b278.nii.gz'))

    Args:
        dwi_images (list of str): the list with the input filenames
        output_fname (str): the output filename
        sort (boolean): if true we natural sort the list of DWI images before we merge them. If false we don't.
            The default is True.
    """
    images = []
    header = None

    if sort:
        def natural_key(_str):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', _str)]
        dwi_images.sort(key=natural_key)

    for dwi_image in dwi_images:
        nib_container = nib.load(dwi_image)
        header = header or nib_container.get_header()
        image_data = nib_container.get_data()

        if len(image_data.shape) < 4:
            image_data = np.expand_dims(image_data, axis=3)

        images.append(image_data)

    combined_image = np.concatenate(images, axis=3)
    nib.Nifti1Image(combined_image, None, header).to_filename(output_fname)
