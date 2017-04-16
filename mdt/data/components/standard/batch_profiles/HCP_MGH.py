import glob
import os
import mdt
from mdt.batch_utils import SimpleBatchProfile, BatchFitProtocolLoader, SimpleSubjectInfo

__author__ = 'Robbert Harms'
__date__ = "2015-07-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

meta_info = {'title': 'HCP MGH',
             'description': '''

The profile for the MGH data from the Human Connectome project.

This assumes that you downloaded and extracted the MGH data in one folder and that you now have one folder per subject.

You can provide the noise standard deviation to use using a noise_std file containing a single float.

Example directory layout:
    /mgh_*/diff/preproc/mri/diff_preproc.nii(.gz)
    /mgh_*/diff/preproc/bvals.txt
    /mgh_*/diff/preproc/bvecs_fsl_moco_norm.txt

Optional items (these will take precedence if present):
    /mgh_*/diff/preproc/diff_preproc.bval
    /mgh_*/diff/preproc/diff_preproc.bvec
    /mgh_*/diff/preproc/diff_preproc.prtcl
    /mgh_*/diff/preproc/diff_preproc_mask.nii(.gz)
    /mgh_*/diff/preproc/mri/diff_preproc_mask.nii(.gz)
    /mgh_*/diff/preproc/noise_std.{txt,nii,nii.gz}
'''}


class HCP_MGH(SimpleBatchProfile):

    def __init__(self, base_directory, **kwargs):
        kwargs['output_base_dir'] = 'diff/preproc/output'
        super(HCP_MGH, self).__init__(base_directory, **kwargs)

    def _get_subjects(self):
        dirs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self._base_directory, '*'))])
        subjects = []
        for subject_id in dirs:
            pjoin = mdt.make_path_joiner(self._base_directory, subject_id, 'diff', 'preproc')
            if os.path.isdir(pjoin()):
                dwi_fname = list(glob.glob(pjoin('mri', 'diff_preproc.nii*')))[0]
                noise_std = self._autoload_noise_std(subject_id, file_path=pjoin('noise_std'))

                bval_fname = pjoin('bvals.txt')
                if os.path.isfile(pjoin('diff_preproc.bval')):
                    bval_fname = pjoin('diff_preproc.bval')

                bvec_fname = pjoin('bvecs_fsl_moco_norm.txt')
                if os.path.isfile(pjoin('diff_preproc.bvec')):
                    bvec_fname = pjoin('diff_preproc.bvec')

                prtcl_fname = None
                if os.path.isfile(pjoin('diff_preproc.prtcl')):
                    prtcl_fname = pjoin('diff_preproc.prtcl')

                mask_fname = None
                if list(glob.glob(pjoin('diff_preproc_mask.nii*'))):
                    mask_fname = list(glob.glob(pjoin('diff_preproc_mask.nii*')))[0]

                if mask_fname is None:
                    if list(glob.glob(pjoin('mri', 'diff_preproc_mask.nii*'))):
                        mask_fname = list(glob.glob(pjoin('mri', 'diff_preproc_mask.nii*')))[0]

                protocol_loader = BatchFitProtocolLoader(
                    pjoin(),
                    protocol_fname=prtcl_fname, bvec_fname=bvec_fname, bval_fname=bval_fname,
                    protocol_columns={'Delta': 21.8e-3, 'delta': 12.9e-3, 'TR': 8800e-3, 'TE': 57e-3})

                output_dir = self._get_subject_output_dir(subject_id, mask_fname)

                subjects.append(SimpleSubjectInfo(subject_id, dwi_fname, protocol_loader, mask_fname, output_dir,
                                                  noise_std=noise_std))
        return subjects

    def __str__(self):
        return meta_info['title']
