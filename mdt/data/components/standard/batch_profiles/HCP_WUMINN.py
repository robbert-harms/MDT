import glob
import os
import mdt
from mdt.batch_utils import SimpleBatchProfile, SimpleSubjectInfo

__author__ = 'Robbert Harms'
__date__ = "2015-07-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

meta_info = {'title': 'HCP WU-Minn',
             'description': '''

The profile for the WU-Minn data from the Human Connectome project',

This assumes that you downloaded and extracted the WU-Minn data in one folder which gives one folder per subject.

You can provide the noise standard deviation to use using a noise_std file containing a single float.

Example directory layout:
    /*/T1w/Diffusion/data.nii.gz
    /*/T1w/Diffusion/bvals
    /*/T1w/Diffusion/bvecs
    /*/T1w/Diffusion/nodif_brain_mask.nii.gz
    /*/T1w/Diffusion/grad_dev.nii.gz

Optional items (these will take precedence if present):
    /*/T1w/Diffusion/data.bval
    /*/T1w/Diffusion/data.bvec
    /*/T1w/Diffusion/data.prtcl
    /*/T1w/Diffusion/data_mask.nii(.gz)
    /*/T1w/Diffusion/noise_std
'''}


class HCP_WUMINN(SimpleBatchProfile):

    def _get_subjects(self, data_folder):
        subjects = []
        for subject_id in sorted([os.path.basename(f) for f in glob.glob(os.path.join(data_folder, '*'))]):
            pjoin = mdt.make_path_joiner(data_folder, subject_id, 'T1w', 'Diffusion')
            if os.path.isdir(pjoin()):
                subject_info = self._get_subject_in_directory(data_folder, subject_id, pjoin)
                if subject_info:
                    subjects.append(subject_info)
        return subjects

    def _get_subject_in_directory(self, data_folder, subject_id, pjoin):
        """Get the information about the given subject in the given directory.

        Args:
            subject_id (str): the id of this subject we are loading
            pjoin (PathJoiner): the path joiner pointing to the directory of that subject

        Returns:
            SimpleSubjectInfo or None: the subject info for this particular subject
        """
        noise_std = self._autoload_noise_std(data_folder, subject_id, file_path=pjoin('noise_std'))

        protocol_loader = self._autoload_protocol(
            pjoin(),
            protocols_to_try=['data.prtcl'],
            bvals_to_try=['data.bval', 'bvals'],
            bvecs_to_try=['data.bvec', 'bvecs'],
            protocol_columns={'Delta': 43.1e-3, 'delta': 10.6e-3, 'TE': 89.5e-3, 'TR': 5520e-3})

        mask_fname = self._get_first_existing_nifti(['data_mask', 'nodif_brain_mask'], prepend_path=pjoin())

        return SimpleSubjectInfo(pjoin(), subject_id, pjoin('data'), protocol_loader, mask_fname,
                                 gradient_deviations=pjoin('grad_dev'), noise_std=noise_std)

    def __str__(self):
        return meta_info['title']
