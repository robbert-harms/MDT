import glob
import os
from mdt.batch_utils import SimpleBatchProfile, BatchFitProtocolLoader, SimpleSubjectInfo

__author__ = 'Robbert Harms'
__date__ = "2015-07-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

meta_info = {'title': 'Directory per subject',
             'description': 'General layout for batch fitting with a folder per subject.',
             'directory_layout':
'''
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
    /*/Delta (in seconds)

    /*/*_mask.nii(.gz)

    /*/noise_std.{txt,nii,nii.gz}

The optional items TE, Delta and delta provide extra information about the protocol.
They should either contain exactly 1 value (for all protocol lines), or a value per protocol line.
'''}


class DirPerSubject(SimpleBatchProfile):

    def __init__(self):
        super(DirPerSubject, self).__init__()
        self.use_gradient_deviations = False

    def _get_subjects(self):
        dirs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self._root_dir, '*'))])
        subjects = []

        for subject_id in dirs:
            niftis = glob.glob(os.path.join(self._root_dir, subject_id, '*.nii*'))
            dwis = list(filter(lambda v: all(name not in v for name in ['_mask', 'grad_dev', 'noise_std']), niftis))
            masks = list(filter(lambda v: '_mask' in v, niftis))
            grad_devs = list(filter(lambda v: 'grad_dev' in v, niftis))
            protocols = glob.glob(os.path.join(self._root_dir, subject_id, '*prtcl'))
            bvals = glob.glob(os.path.join(self._root_dir, subject_id, '*bval*'))
            bvecs = glob.glob(os.path.join(self._root_dir, subject_id, '*bvec*'))
            noise_std = self._autoload_noise_std(subject_id)

            if dwis:
                dwi_fname = dwis[0]

                mask_first_choice = glob.glob(os.path.join(self._root_dir, subject_id, 'mask.nii*'))
                if mask_first_choice:
                    mask_fname = mask_first_choice[0]
                else:
                    mask_fname = masks[0] if masks else None

                if not self.use_gradient_deviations:
                    grad_dev = None
                else:
                    grad_dev = grad_devs[0] if grad_devs else None

                protocol_fname = protocols[0] if protocols else None
                bval_fname = bvals[0] if bvals else None
                bvec_fname = bvecs[0] if bvecs else None

                if dwi_fname and (protocol_fname or (bval_fname and bvec_fname)):
                    protocol_loader = BatchFitProtocolLoader(
                        os.path.join(self._root_dir, subject_id),
                        protocol_fname=protocol_fname, bvec_fname=bvec_fname,
                        bval_fname=bval_fname)

                    output_dir = self._get_subject_output_dir(subject_id, mask_fname)

                    subjects.append(SimpleSubjectInfo(subject_id, dwi_fname, protocol_loader, mask_fname, output_dir,
                                                      gradient_deviations=grad_dev, noise_std=noise_std))

        return subjects

    def __str__(self):
        return meta_info['title']
