import glob
import os

import mdt
from mdt.batch_utils import SimpleBatchProfile, BatchFitProtocolLoader, SimpleSubjectInfo

__author__ = 'Robbert Harms'
__date__ = "2015-07-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

meta_info = {'title': 'Directory per subject',
             'description': '''

General layout for batch fitting with a folder per subject.

Every subject has its own directory. For every type of file (protocol, bvec, bval, TE, Delta, delta, DWI, mask)
we use the first one found.

You can provide the noise standard deviation to use using a noise_std file containing a single float.

Example directory layout:
    /a/*.nii(.gz)
    /a/*bval*
    /a/*bvec*

    /b/...
    /c/...

Optional items:
    /*/prtcl
    /*/*.prtcl

    /*/TE (in seconds)
    /*/delta (in seconds)
    /*/Delta (in seconds) (it is also possible to use a filename /*/big_delta for case insensitive filesystems)

    /*/*_mask.nii(.gz)

    /*/noise_std.{txt,nii,nii.gz}

The optional items TE, Delta and delta provide extra information about the protocol.
They should either contain exactly 1 value (for all protocol lines), or a value per protocol line.
'''}


class DirPerSubject(SimpleBatchProfile):

    def __init__(self, base_directory, use_gradient_deviations=False, **kwargs):
        super(DirPerSubject, self).__init__(base_directory, **kwargs)
        self.use_gradient_deviations = use_gradient_deviations
        self._constructor_kwargs.update(use_gradient_deviations=self.use_gradient_deviations)

    def _get_subjects(self):
        subjects = []
        for subject_id in sorted([os.path.basename(f) for f in glob.glob(os.path.join(self._base_directory, '*'))]):
            pjoin = mdt.make_path_joiner(self._base_directory, subject_id)
            subject_info = self._get_subject_in_directory(subject_id, pjoin)
            if subject_info:
                subjects.append(subject_info)
        return subjects

    def _get_subject_in_directory(self, subject_id, pjoin):
        """Get the information about the given subject in the given directory.

        Args:
            subject_id (str): the id of this subject we are loading
            pjoin (PathJoiner): the path joiner pointing to the directory of that subject

        Returns:
            SimpleSubjectInfo or None: the subject info for this particular subject
        """
        niftis = glob.glob(pjoin('*.nii*'))
        dwis = list(filter(lambda v: all(name not in v for name in ['_mask', 'grad_dev', 'noise_std']), niftis))
        masks = list(filter(lambda v: '_mask' in v, niftis))
        grad_devs = list(filter(lambda v: 'grad_dev' in v, niftis))
        protocols = glob.glob(pjoin('*prtcl'))
        bvals = glob.glob(pjoin('*bval*'))
        bvecs = glob.glob(pjoin('*bvec*'))
        noise_std = self._autoload_noise_std(subject_id)

        if dwis:
            dwi_fname = dwis[0]

            mask_fnames = [pjoin('mask.nii')]
            mask_fnames.extend(masks)
            mask_fname = self._get_first_existing_nifti(mask_fnames)

            grad_dev = grad_devs[0] if grad_devs else None

            protocol_fname = protocols[0] if protocols else None
            bval_fname = bvals[0] if bvals else None
            bvec_fname = bvecs[0] if bvecs else None

            if dwi_fname and (protocol_fname or (bval_fname and bvec_fname)):
                protocol_loader = BatchFitProtocolLoader(
                    os.path.join(self._base_directory, subject_id),
                    protocol_fname=protocol_fname, bvec_fname=bvec_fname,
                    bval_fname=bval_fname)

                output_dir = self._get_subject_output_dir(subject_id, mask_fname)

                return SimpleSubjectInfo(subject_id, dwi_fname, protocol_loader, mask_fname, output_dir,
                                         gradient_deviations=grad_dev,
                                         use_gradient_deviations=self.use_gradient_deviations,
                                         noise_std=noise_std)
        return None

    def __str__(self):
        return meta_info['title']
