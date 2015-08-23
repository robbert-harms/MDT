import glob
import os
import mdt
from mdt.batch_utils import SimpleBatchProfile, SimpleProtocolLoader, SimpleSubjectInfo

__author__ = 'Robbert Harms'
__date__ = "2015-07-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

meta_info = {'title': 'HCP MGH',
             'description': 'The profile for the MGH data from the Human Connectome project',
             'directory_layout':
'''
This assumes that you downloaded and extracted the MGH data in one folder and that you now have one folder per subject.

Example directory layout:
    /mgh_*/diff/preproc/mri/diff_preproc.nii(.gz)
    /mgh_*/diff/preproc/bvals.txt
    /mgh_*/diff/preproc/bvecs_fsl_moco_norm.txt

Optional items (these will take precedence if present):
    /mgh_*/diff/preproc/diff_preproc.bval
    /mgh_*/diff/preproc/diff_preproc.bvec
    /mgh_*/diff/preproc/diff_preproc.prtcl
    /mgh_*/diff/preproc/diff_preproc_mask.nii(.gz)
'''}

class HCP_MGH_Profile(SimpleBatchProfile):

    def _get_subjects(self):
        dirs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self._root_dir, '*'))])
        subjects = []
        for subject_id in dirs:
            pjoin = mdt.make_path_joiner(self._root_dir, subject_id, 'diff', 'preproc')
            if os.path.isdir(pjoin()):
                dwi_fname = glob.glob(pjoin('mri', 'diff_preproc.nii*'))[0]

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
                if glob.glob(pjoin('diff_preproc_mask.nii*')):
                    mask_fname = glob.glob(pjoin('diff_preproc_mask.nii*'))[0]

                protocol_loader = SimpleProtocolLoader(
                    prtcl_fname=prtcl_fname, bvec_fname=bvec_fname, bval_fname=bval_fname,
                    extra_cols={'Delta': 12.9e-3, 'delta': 21.8e-3, 'TR': 8800e-3, 'TE': 57e-3})

                output_dir = os.path.join(self._root_dir, subject_id, 'diff', 'preproc', 'output')

                subjects.append(SimpleSubjectInfo(subject_id, dwi_fname, protocol_loader, mask_fname, output_dir))
        return subjects

    def __str__(self):
        return meta_info['title']