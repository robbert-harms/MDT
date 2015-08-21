import glob
import os
import mdt
from mdt.utils import SimpleBatchProfile

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

    def get_batch_fit_config_options(self):
        return {'protocol': {'extra_columns': {'Delta': 12.9e-3, 'delta': 21.8e-3, 'TR': 8800e-3, 'TE': 57e-3}}}

    def get_output_directory(self, root_dir, subject_id):
        return os.path.join(root_dir, subject_id, 'diff', 'preproc', 'output')

    def _get_subjects(self, root_dir):
        dirs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(root_dir, '*'))])
        subjects = []
        for d in dirs:
            info = {}
            pjoin = mdt.make_path_joiner(root_dir, d, 'diff', 'preproc')
            if os.path.isdir(pjoin()):
                if glob.glob(pjoin('mri', 'diff_preproc.nii*')):
                    info['dwi'] = glob.glob(pjoin('mri', 'diff_preproc.nii*'))[0]

                if os.path.isfile(pjoin('diff_preproc.bval')):
                    info['bval'] = pjoin('diff_preproc.bval')
                elif os.path.isfile(pjoin('bvals.txt')):
                    info['bval'] = pjoin('bvals.txt')

                if os.path.isfile(pjoin('diff_preproc.bvec')):
                    info['bvec'] = pjoin('diff_preproc.bvec')
                elif os.path.isfile(pjoin('bvecs_fsl_moco_norm.txt')):
                    info['bvec'] = pjoin('bvecs_fsl_moco_norm.txt')

                if os.path.isfile(pjoin('diff_preproc.prtcl')):
                    info['prtcl'] = pjoin('diff_preproc.prtcl')

                if glob.glob(pjoin('diff_preproc_mask.nii*')):
                    info['mask'] = glob.glob(pjoin('diff_preproc_mask.nii*'))[0]

            if 'dwi' in info and (('bval' in info and 'bvec' in info) or 'prtcl' in info):
                subjects.append((d, info))
        return subjects

    def __repr__(self):
        return meta_info['title']