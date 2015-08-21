import logging
import os
import shutil
import timeit
import pickle
import time
from six import string_types
from mdt.protocols import load_bvec_bval, load_from_protocol, write_protocol
from mdt.components_loader import get_model
from mdt.cascade_model import CascadeModelInterface
from mdt import configuration
from mdt.IO import Nifti
from mdt.utils import create_roi, configure_per_model_logging, restore_volumes, recursive_merge_dict, \
    load_problem_data, ProtocolProblemError, MetaOptimizerBuilder, get_cl_devices, \
    get_model_config, apply_model_protocol_options, model_output_exists, get_batch_profile
from mdt.masking import create_write_median_otsu_brain_mask
from mot import runtime_configuration

__author__ = 'Robbert Harms'
__date__ = "2015-05-01"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BatchFitting(object):

    def __init__(self, data_folder, batch_profile=None, subjects=None, recalculate=False, cl_device_ind=None):
        """This class is meant to make running computations as simple as possible.

        The idea is that a single folder is enough to fit_model the computations.
        For configuration of the optimizers uses the users configuration file. For batch fitting specific options use
        the options parameter.

        One can optionally give it the batch_profile to use for the fitting. If not given, this class
        will attempt to use the batch_profile that fits the data folder best.

        The general optimization options are loaded in this order:
            0) default options
            1) options from the batch profile

        Setting the cl_device_ind has the side effect that it changes the current run time cl_device settings in the
        PPPE toolkit.

        Args:
            data_folder (str): the main directory to look for items to process.
            batch_profile (SubjectCrawler or str): the subjects crawler to use to get all the subjects, can
                also be the name of the subject crawler to load from the users folder.
            subjects (list of int): either a list of subjects to process or the index of a single subject to process.
                This indexes the list of subjects returned by the batch profile.
            recalculate (boolean): If we want to recalculate the results if they are already present.
            cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
                get_cl_devices().
        """
        self._data_folder = data_folder
        self._config = configuration.config['batch_fitting']
        self._logger = logging.getLogger(__name__)
        self._batch_profile = get_batch_profile(batch_profile, self._data_folder)
        self._cl_device_ind = cl_device_ind
        self._subjects = subjects
        self._recalculate = recalculate

        if self._batch_profile is None:
            raise RuntimeError('No suitable batch profile could be '
                               'found for the directory {0}'.format(os.path.abspath(self._data_folder)))

        self._config = recursive_merge_dict(self._config, self._batch_profile.get_batch_fit_config_options())

        self._logger.info('Using batch profile: {0}'.format(self._batch_profile))
        self._subjects = self._get_subjects()

        self._logger.info('Subjects found: {0}'.format(self._batch_profile.get_subjects_count(self._data_folder)))

        if self._cl_device_ind is not None:
            runtime_configuration.runtime_config['cl_environments'] = [get_cl_devices()[self._cl_device_ind]]

    def get_all_subjects_info(self):
        """Get a dictionary with the info of all the found subjects.

        This will return information about all the subjects found and will disregard parameter 'subjects'
        that limits the amount of subjects we will run.

        Returns:
            A listing of all the found subjects from the batch profile. This will look like:
                [('<subject_name>', {<info dict>}), (...), ...]
        """
        return self._batch_profile.get_subjects(self._data_folder)

    def get_subjects_info(self):
        """Get a dictionary with the info of the subject we will run computations on.

        This will return information about only the subjects that we will use in the batch fitting.

        Returns:
            A listing of all the found subjects from the batch profile. This will look like:
                [('<subject_name>', {<info dict>}), (...), ...]
        """
        return self._subjects

    def run(self):
        """Run the computations on the current dir with all the configured options. """
        self._logger.info('Running computations on {0} subjects'.format(len(self._subjects)))

        batch_items = [{'subject': s,
                        'output_dir': self._batch_profile.get_output_directory(self._data_folder, s[0])}
                       for s in self._subjects]

        run_func = _BatchFitRunner(self._config, self._recalculate, self._cl_device_ind)
        map(run_func, batch_items)

        return self._subjects

    def _get_subjects(self):
        subjects = self._batch_profile.get_subjects(self._data_folder)
        if self._subjects is not None:
            if hasattr(self._subjects, '__iter__'):
                subjects_selection = self._subjects
            else:
                subjects_selection = [self._subjects]

            returned_subjects = []
            for subject_ind in subjects_selection:
                if 0 <= subject_ind < len(subjects):
                    returned_subjects.append(subjects[subject_ind])
                else:
                    logging.info('The specified subject (in config "subjects") with index number {0} '
                                 'does not exist'.format(subject_ind))
            return returned_subjects
        return subjects


class _BatchFitRunner(object):

    def __init__(self, batch_fitting_config, recalculate, cl_device_ind):
        self._batch_fitting_config = batch_fitting_config
        self._recalculate = recalculate
        self._cl_device_ind = cl_device_ind

    def __call__(self, batch_instance):
        """Run the batch fitting on the given subject.

        This is a module level function to allow for python multiprocessing to work.

        Args:
            batch_instance (dict): contains the items: 'subject', 'config', 'output_dir'
        """
        logger = logging.getLogger(__name__)

        subject = batch_instance['subject']
        output_dir = batch_instance['output_dir']

        protocol = self._get_protocol(subject[0], subject[1])
        brain_mask = self._get_mask(subject[0], subject[1], protocol, output_dir)

        if all(model_output_exists(model, os.path.join(output_dir, self._get_basename(brain_mask)))
               for model in self._batch_fitting_config['models']):
            logger.info('Skipping subject {0}, output exists'.format(subject[0]))
            return

        logger.info('Loading the data (DWI, mask and protocol) of subject {0}'.format(subject[0]))
        problem_data = load_problem_data(subject[1]['dwi'], protocol, brain_mask)

        write_protocol(protocol, os.path.join(output_dir, 'auto_generated_protocol.prtcl'))

        start_time = timeit.default_timer()
        for model in self._batch_fitting_config['models']:
            logger.info('Going to fit model {0} on subject {1}'.format(model, subject[0]))
            try:
                model_fit = ModelFit(model,
                                     problem_data,
                                     os.path.join(output_dir, self._get_basename(brain_mask)),
                                     recalculate=self._recalculate,
                                     only_recalculate_last=False,
                                     model_protocol_options=self._batch_fitting_config['model_protocol_options'],
                                     cl_device_ind=self._cl_device_ind)
                model_fit.run()
            except ProtocolProblemError as ex:
                logger.info('Could not fit model {0} on subject {1} '
                            'due to protocol problems. {2}'.format(model, subject[0], ex))
            else:
                logger.info('Done fitting model {0} on subject {1}'.format(model, subject[0]))
        logger.info('Fitted all models on subject {0} in time {1} (h:m:s)'.format(
            subject[0], time.strftime('%H:%M:%S', time.gmtime(timeit.default_timer() - start_time))))

    def _get_basename(self, f):
        """Get the basename of a file. That is, the name of the files with all extensions stripped."""
        bn = os.path.basename(f)
        bn = bn.replace('.nii.gz', '')
        exts = os.path.splitext(bn)
        if len(exts) > 0:
            return exts[0]
        return bn

    def _get_mask(self, subject_id, subject_info, protocol, output_dir):
        """Helper function for _batch_fit_run()

        Args:
            subject_id (str): the id of this subject
            subject_info (dict): information items about that subject. Contains things like 'dwi', 'bval', 'bvec',
                'prtcl' and 'mask'
            protocol (Protocol): the protocol object
            output_dir (str): if we need to create a new mask, this is the location to write it to
            batch_fitting_config: the configuration of the batch fitting procedure

        Returns:
            str: the filename of the mask to load
        """
        logger = logging.getLogger(__name__)

        if 'mask' in subject_info:
            return subject_info['mask']
        else:
            output_fname = os.path.join(output_dir, 'brain_mask.nii.gz')
            if not os.path.isfile(output_fname):
                logger.info('Creating a brain mask for subject {0}'.format(subject_id))
                create_write_median_otsu_brain_mask(subject_info['dwi'], protocol, output_fname)
            return output_fname

    def _get_protocol(self, subject_id, subject_info):
        """Helper function for _batch_fit_run()

        Args:
            subject_id (str): the id of this subject
            subject_info (dict): information items about that subject. Contains things like 'dwi', 'bval', 'bvec',
                'prtcl', 'TE', 'Delta', 'delta' and 'mask'
            batch_fitting_config: the configuration of the batch fitting procedure
        """
        logger = logging.getLogger(__name__)

        if 'prtcl' in subject_info:
            protocol = load_from_protocol(subject_info['prtcl'])
        else:
            protocol = load_bvec_bval(subject_info['bvec'], subject_info['bval'])

        if self._batch_fitting_config['protocol']['max_G'] is not None:
            protocol.max_G = self._batch_fitting_config['protocol']['max_G']

        for col in self._batch_fitting_config['protocol']['extra_columns'].items():
            protocol.add_column(*col)

        for col in ('TE', 'Delta', 'delta'):
            if col in subject_info:
                protocol.add_column_from_file(col, subject_info[col])

        for column in ['G', 'Delta', 'delta']:
            if not protocol.has_unestimated_column(column):
                logger.debug('Protocol column {column} will be '
                             'estimated for subject {subject}. '
                             'This may give inaccurate results.'.format(column=column, subject=subject_id))
        return protocol


class ModelFit(object):

    def __init__(self, model, problem_data, output_folder, optimizer=None,
                 recalculate=False, only_recalculate_last=False, model_protocol_options=None,
                 cl_device_ind=None):
        """Setup model fitting for the given input model and data.

        This does nothing by itself, please call fit_model() to actually fit_model the optimizer to fit the model.

        Args:
            model (AbstractModel): An implementation of an AbstractModel that contains the model we want to optimize.
            problem_data (ProblemData): the problem data object which contains the dwi image, the dwi header, the
                brain_mask and the protocol to use.
            output_folder (string): The full path to the folder where to place the output
            optimizer (AbstractOptimizer): The optimization routine to use. If None, we create one using the
                configuration files.
            recalculate (boolean): If we want to recalculate the results if they are already present.
            only_recalculate_last (boolean):
                This is only of importance when dealing with CascadeModels.
                If set to true we only recalculate the last element in the chain
                    (if recalculate is set to True, that is).
                If set to false, we recalculate everything. This only holds for the first level of the cascade.
            model_protocol_options (dict): specific model protocol options to use during fitting.
                This is for example used during batch fitting to limit the protocol for certain models.
                For instance, in the Tensor model we generally only want to use the lower b-values.
            cl_device_ind (int): the index of the CL device to use. The index is from the list from the function
                get_cl_devices().
        """
        if isinstance(model, string_types):
            model = get_model(model)

        self._model = model
        self._problem_data = problem_data
        self._output_folder = output_folder
        self._optimizer = optimizer
        self._recalculate = recalculate
        self._only_recalculate_last = only_recalculate_last
        self._model_protocol_options = model_protocol_options
        self._logger = logging.getLogger(__name__)
        self._cl_device_ind = cl_device_ind

        if not model.is_protocol_sufficient(self._problem_data.protocol):
            raise ProtocolProblemError(
                'The given protocol is insufficient for this model. '
                'The reported errors where: {}'.format(self._model.get_protocol_problems(self._problem_data.protocol)))

        if self._cl_device_ind is not None:
            runtime_configuration.runtime_config['cl_environments'] = [get_cl_devices()[self._cl_device_ind]]

    @classmethod
    def load_from_basic_data(cls, model, dwi_info, protocol, brain_mask, output_folder, **kwargs):
        """This will automatically create the problem data object the constructor needs.

        It is better to use the constructor directly since reusing a problem_data object will save you runtime and
        memory if you are optimizing multiple models after each other.

        Still, for ease of use, this method is here.

        Args:
            model (AbstractModel): An implementation of an AbstractModel that contains the model we want to optimize.
            dwi_info (str or list): Either a filename of a DWI image to load, or a list with as first element the image
                and as second a nifti header.
            protocol (str or Protocol): either a filename of a protocol to load, or a Protocol object
            brain_mask (str or ndarray): either a filename of a brain mask or a ndarray which should function as a mask.
            output_folder (string): The full path to the folder where to place the output
            **kwargs (dict): see the constructor

        Returns:
            An instance of this class with all the correct arguments.
        """
        problem_data = load_problem_data(dwi_info, protocol, brain_mask)
        return cls(model, problem_data, output_folder, **kwargs)

    def run(self):
        """Run the model and return the resulting maps

        If we will not recalculate and the maps already exists, we will load the maps from file and return those.

        Returns:
            The result maps for the model we are running.
        """
        self._run(self._model, self._recalculate, self._only_recalculate_last, {})

    def _run(self, model, recalculate, only_recalculate_last, meta_optimizer_config):
        """Recursively calculate the (cascade) models"""
        if isinstance(model, CascadeModelInterface):
            results = {}
            last_result = None
            while model.has_next():
                sub_model = model.get_next(results)
                meta_optimizer_config = model.get_optimization_config(sub_model)

                sub_recalculate = False
                if recalculate:
                    if only_recalculate_last:
                        if not model.has_next():
                            sub_recalculate = True
                    else:
                        sub_recalculate = True

                new_results = self._run(sub_model, sub_recalculate, recalculate, meta_optimizer_config)
                results.update({sub_model.name: new_results})
                last_result = new_results

            model.reset()
            return last_result

        return self._run_single_model(model, recalculate, meta_optimizer_config)

    def _run_single_model(self, model, recalculate, meta_optimizer_config):
        self._logger.info('Preparing for model {0}'.format(model.name))
        optimizer = self._optimizer or MetaOptimizerBuilder(meta_optimizer_config).construct(model.name)

        if self._cl_device_ind is not None:
            optimizer.cl_environments = [get_cl_devices()[self._cl_device_ind]]

        model_protocol_options = get_model_config(model.name, self._model_protocol_options)
        problem_data = apply_model_protocol_options(model_protocol_options, self._problem_data)
        return fit_single_model(model, problem_data, self._output_folder, optimizer, recalculate=recalculate)


def fit_single_model(model, problem_data, output_folder, optimizer, recalculate=False):
    """Fits a single model.

    This does not accept cascade models. Please use the more general ModelFit class for single and cascade models.

    Args:
        model (AbstractModel): An implementation of an AbstractModel that contains the model we want to optimize.
        problem_data (DMRIProblemData): The problem data object with which the model is initialized before running
        output_folder (string): The full path to the folder where to place the output
        optimizer (AbstractOptimizer): The optimization routine to use.
        recalculate (boolean): If we want to recalculate the results if they are already present.
    """
    output_path = os.path.join(output_folder, model.name)
    logger = logging.getLogger(__name__)
    model.set_problem_data(problem_data)

    if not model.is_protocol_sufficient(problem_data.protocol):
        raise ProtocolProblemError(
            'The given protocol is insufficient for this model. '
            'The reported errors where: {}'.format(model.get_protocol_problems(problem_data.protocol)))

    if recalculate:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
    else:
        if model_output_exists(model, output_folder):
            maps = Nifti.read_volume_maps(output_path)
            logger.info('Not recalculating {} model'.format(model.name))
            return create_roi(maps, problem_data.mask)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    configure_per_model_logging(output_path)

    minimize_start_time = timeit.default_timer()
    logger.info('Fitting {} model'.format(model.name))

    results, other_output = optimizer.minimize(model, full_output=True)

    _write_output(results, other_output, problem_data, output_path)

    run_time = timeit.default_timer() - minimize_start_time
    run_time_str = time.strftime('%H:%M:%S', time.gmtime(run_time))
    logger.info('Fitted {0} model with runtime {1} (h:m:s).'.format(model.name, run_time_str))
    configure_per_model_logging(None)

    return results


def _write_output(results, other_output, problem_data, output_path):
    volume_maps = restore_volumes(results, problem_data.mask)
    Nifti.write_volume_maps(volume_maps, output_path, problem_data.volume_header)

    for k, v in other_output.items():
        with open(os.path.join(output_path, k + '.pyobj'), 'wb') as f:
            pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)