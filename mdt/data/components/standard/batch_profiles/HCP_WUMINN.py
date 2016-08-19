import glob
import os
import mdt
from mdt.batch_utils import SimpleBatchProfile, BatchFitProtocolLoader, SimpleSubjectInfo

__author__ = 'Robbert Harms'
__date__ = "2015-07-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

meta_info = {'title': 'HCP WU-Minn',
             'description': 'The profile for the WU-Minn data from the Human Connectome project',
             'directory_layout':
'''
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

    def __init__(self):
        super(HCP_WUMINN, self).__init__()
        self.use_gradient_deviations = False
        self._output_base_dir = 'T1w/Diffusion/output'
        self._append_mask_name_to_output_sub_dir = False
        self.models_to_fit = ('BallStick (Cascade)',
                              'Tensor (Cascade)',
                              'Noddi (Cascade|fixed)',
                              'Charmed_r1 (Cascade|fixed)',
                              'Charmed_r2 (Cascade|fixed)',
                              'Charmed (Cascade|fixed)')

    def _get_subjects(self):
        dirs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self._root_dir, '*'))])
        subjects = []
        for subject_id in dirs:
            pjoin = mdt.make_path_joiner(self._root_dir, subject_id, 'T1w', 'Diffusion')
            if os.path.isdir(pjoin()):
                dwi_fname = list(glob.glob(pjoin('data.nii*')))[0]

                noise_std = self._autoload_noise_std(subject_id, file_path=pjoin('noise_std'))

                bval_fname = pjoin('bvals')
                if os.path.isfile(pjoin('data.bval')):
                    bval_fname = pjoin('data.bval')

                bvec_fname = pjoin('bvecs')
                if os.path.isfile(pjoin('data.bvec')):
                    bvec_fname = pjoin('data.bvec')

                prtcl_fname = None
                if os.path.isfile(pjoin('data.prtcl')):
                    prtcl_fname = pjoin('data.prtcl')

                mask_fname = list(glob.glob(pjoin('nodif_brain_mask.nii*')))[0]
                if list(glob.glob(pjoin('data_mask.nii*'))):
                    mask_fname = list(glob.glob(pjoin('data_mask.nii*')))[0]

                if self.use_gradient_deviations:
                    grad_dev = pjoin('grad_dev.nii.gz')
                else:
                    grad_dev = None

                protocol_loader = BatchFitProtocolLoader(
                    pjoin(),
                    protocol_fname=prtcl_fname, bvec_fname=bvec_fname,
                    bval_fname=bval_fname, protocol_options={'Delta': 43.1e-3, 'delta': 10.6e-3,
                                                             'TE': 89.5e-3, 'TR': 5520e-3})

                output_dir = self._get_subject_output_dir(subject_id, mask_fname)

                subjects.append(SimpleSubjectInfo(subject_id, dwi_fname, protocol_loader, mask_fname,
                                                  output_dir, gradient_deviations=grad_dev, noise_std=noise_std))
        return subjects

    def __str__(self):
        return meta_info['title']
