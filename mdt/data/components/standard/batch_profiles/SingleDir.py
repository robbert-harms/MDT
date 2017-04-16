import glob
import os
import mdt
from mdt.batch_utils import SimpleBatchProfile, BatchFitProtocolLoader, SimpleSubjectInfo
from mdt.utils import split_image_path

__author__ = 'Robbert Harms'
__date__ = "2015-07-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

meta_info = {'title': 'Single directory',
             'description': '''

All files in one directory, distinct name per subject.

For every subject in the directory the base name must be equal to be recognized as
belonging to one subject.

Example directory layout:
    /a.nii.gz
    /a.bval
    /a.bvec

    /b.nii
    /b.prtcl

    /c.nii.gz
    /c.prtcl
    /c_mask.nii.gz

This gives three subjects, 'a', 'b' and 'c' with a Diffusion Weighted Image recognized by the extension .nii(.gz) and
a bvec and bval or protocol file for the protocol information. Subject 'c' also has a brain mask associated.

If there is a file mask.nii(.gz) present in the directory it is used as the default mask for all the subjects
in the single directory that do not have their own mask.

You can provide the noise standard deviation to use using a noise_std file (per subject). This should either
be a .txt file (global noise) or a nifti file (voxel wise noise).

Optional items (all case sensitive):
    /<name>.TE
    /<name>.Delta (can also be named /<name>.big_delta for case insensitive filesystems)
    /<name>.delta
    /<name>.noise_std.{txt,nii,nii.gz}
'''}


class SingleDir(SimpleBatchProfile):

    def _get_subjects(self):
        pjoin = mdt.make_path_joiner(self._base_directory)

        files = [os.path.basename(f) for f in glob.glob(pjoin('*'))]
        basenames = sorted(list({split_image_path(f)[1] for f in files}))
        subjects = []

        protocol_options = ['TE', 'TR', 'Delta', 'delta', 'maxG']

        default_mask = None
        if list(glob.glob(pjoin('mask.nii*'))):
            default_mask = list(glob.glob(pjoin('mask.nii*')))[0]

        for basename in basenames:
            dwi_fname = None
            if basename + '.nii' in files:
                dwi_fname = pjoin(basename + '.nii')
            elif basename + '.nii.gz' in files:
                dwi_fname = pjoin(basename + '.nii.gz')
            elif basename + '.hdr' in files and basename + '.img' in files:
                dwi_fname = pjoin(basename + '.hdr')

            noise_std = self._autoload_noise_std(basename, file_path=pjoin(basename + '.noise_std'))

            prtcl_fname = None
            if basename + '.prtcl' in files:
                prtcl_fname = pjoin(basename + '.prtcl')

            bval_fname = None
            if basename + '.bval' in files:
                bval_fname = pjoin(basename + '.bval')

            bvec_fname = None
            if basename + '.bvec' in files:
                bvec_fname = pjoin(basename + '.bvec')

            mask_fname = default_mask
            if basename + '_mask.nii' in files:
                mask_fname = pjoin(basename + '_mask.nii')
            elif basename + '_mask.nii.gz' in files:
                mask_fname = pjoin(basename + '_mask.nii.gz')

            extra_cols_from_file = {}
            for option in protocol_options:
                if basename + '.' + option in files:
                    extra_cols_from_file.update({option: pjoin(basename + '.' + option)})

            if basename + '.big_delta' in files:
                extra_cols_from_file.update({'Delta': pjoin(basename + '.' + 'big_delta')})

            if dwi_fname and (prtcl_fname or (bval_fname and bvec_fname)):
                protocol_loader = BatchFitProtocolLoader(
                    pjoin(),
                    protocol_fname=prtcl_fname, bvec_fname=bvec_fname, bval_fname=bval_fname,
                    protocol_columns=extra_cols_from_file)

                output_dir = self._get_subject_output_dir(basename, mask_fname, pjoin(self.output_base_dir, basename))

                subjects.append(SimpleSubjectInfo(basename, dwi_fname, protocol_loader, mask_fname, output_dir,
                                                  noise_std=noise_std))
        return subjects

    def __str__(self):
        return meta_info['title']
