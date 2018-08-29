import glob
import os
from itertools import filterfalse

from mdt.lib.batch_utils import SimpleBatchProfile, BatchFitProtocolLoader, SimpleSubjectInfo
from mdt.component_templates.base import ComponentBuilder, ComponentTemplate

__author__ = 'Robbert Harms'
__date__ = "2017-02-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class BatchProfileBuilder(ComponentBuilder):

    def _create_class(self, template):
        """Creates classes with as base class SimpleBatchProfile

        Args:
            template (BatchProfileTemplate): the batch profile config template

        Returns:
            SimpleBatchProfile: an subclass of a batch profile
        """
        class AutoCreatedBatchProfile(SimpleBatchProfile):
            def _get_subjects(self, data_folder):
                dirs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(data_folder, '*'))])

                subjects = []
                for subject_id in dirs:
                    subject_base_folder = os.path.join(
                        data_folder, template.subject_base_folder.format(subject_id=subject_id))

                    def _prepare_path(template_path):
                        if template_path is None:
                            return None
                        return template_path.format(data_folder=data_folder,
                                                    subject_id=subject_id,
                                                    subject_base_folder=subject_base_folder)

                    data_glob = glob.glob(_prepare_path(template.data_fname))
                    if not list(data_glob):
                        break

                    noise_std = self._autoload_noise_std(data_folder, subject_id, file_pattern=template.noise_std_fname)

                    protocol_loader = BatchFitProtocolLoader(
                        _prepare_path(template.protocol_auto_dir),
                        protocol_fname=_prepare_path(template.protocol_fname),
                        bvec_fname=_prepare_path(template.bvec_fname),
                        bval_fname=_prepare_path(template.bval_fname),
                        protocol_columns=template.protocol_columns)

                    mask_fname = None
                    if list(glob.glob(_prepare_path(template.mask_fname))):
                        mask_fname = glob.glob(_prepare_path(template.mask_fname))[0]
                        data_glob = list(filterfalse(lambda v: v == mask_fname, data_glob))

                    grad_dev = None
                    if list(glob.glob(_prepare_path(template.gradient_deviations_fname))):
                        grad_dev = glob.glob(_prepare_path(template.gradient_deviations_fname))[0]
                        data_glob = list(filterfalse(lambda v: v == grad_dev, data_glob))

                    subjects.append(SimpleSubjectInfo(
                        subject_base_folder, subject_id, data_glob[0],
                        protocol_loader, mask_fname, noise_std=noise_std,
                        gradient_deviations=grad_dev))

                return subjects

            def __str__(self):
                return template.name

        for name, method in template.bound_methods.items():
            setattr(AutoCreatedBatchProfile, name, method)

        return AutoCreatedBatchProfile


class BatchProfileTemplate(ComponentTemplate):
    """The batch profile template to inherit.

    Attributes:
        name (str): the name of this batch profile
        description (str): the description
        subject_base_folder (str): the base folder for this subject. Allows expansion of ``{subject_id}``.
        data_fname (str): the filename of the data volumes file. Allows expansion of ``{subject_id}``
            and ``{subject_base_folder}``, and supports globbing. Results are afterwards filtered to exclude matches
            of the mask and gradient deviations files.
        mask_fname (str): the filename of the mask. Allows expansion of ``{subject_id}`` and ``{subject_base_folder}``,
            and supports globbing.
        noise_std_fname (str): the filename of the noise standard deviation file. Can be textfile or a nifti file.
            Allows expansion of ``{subject_id}`` and ``{subject_base_folder}``, and supports globbing.
        gradient_deviations_fname (str): the filename of the gradient deviations. Allows expansion of
            ``{subject_id}`` and ``{subject_base_folder}``, and supports globbing.
        protocol_auto_dir (str): the directory from which MDT will try to autoload a directory.
            Supports ``{subject_id}`` and ``{subject_base_folder}``, and supports globbing.
        protocol_fname (str): the filename of the protocol file to use. Supports ``{subject_id}``
            and ``{subject_base_folder}``, and supports globbing. If provided, we use it instead of the automatically
            searched default.
        bvec_fname (str): the filename of the bvec file to use. Supports ``{subject_id}``
            and ``{subject_base_folder}``, and supports globbing. If provided, we use it instead of the automatically
            searched default.
        bval_fname (str): the filename of the bval file to use. Supports ``{subject_id}``
            and ``{subject_base_folder}``, and supports globbing. If provided, we use it instead of the automatically
            searched default.
        protocol_columns (dict): a dictionary with additional columns to add to the protocol file. Use this for
            default values for all subjects in your study.
    """
    _component_type = 'batch_profiles'
    _builder = BatchProfileBuilder()

    name = None
    description = ''

    subject_base_folder = '{subject_id}'

    data_fname = '{subject_base_folder}/*.nii*'
    mask_fname = '{subject_base_folder}/*mask.nii*'
    noise_std_fname = '{subject_base_folder}/noise_std*'
    gradient_deviations_fname = '{subject_base_folder}/grad_dev.nii*'

    protocol_auto_dir = '{subject_base_folder}'
    protocol_fname = None
    bvec_fname = None
    bval_fname = None
    protocol_columns = {}
