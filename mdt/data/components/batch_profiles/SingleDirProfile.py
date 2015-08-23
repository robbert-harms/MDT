import glob
import os
from mdt.batch_utils import BatchSubjectInfo, SimpleBatchProfile

__author__ = 'Robbert Harms'
__date__ = "2015-07-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

meta_info = {'title': 'Single directory',
             'description': 'All files in one directory, distinct name per subject.',
             'directory_layout':
'''
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

If there is a file mask.nii(.gz) it is used as the mask for all the subjects in the single directory.

Optional items:
    /<name>.TE (case sensitive)
    /<name>.Delta (case sensitive)
    /<name>.delta (case sensitive)

These provide extra information about the protocol. They should either contain exactly 1 value (for all protocol lines),
or a value per protocol line.
'''}

class SingleDirProfile(SimpleBatchProfile):

    def get_output_directory(self, subject_id):
        return os.path.join('output', subject_id)

    def _get_subjects(self):
        files = [os.path.basename(f) for f in glob.glob(os.path.join(self._root_dir, '*'))]
        basenames = sorted(list({self._get_basename(f) for f in files}))
        subjects = []

        files_to_look_for = [('bval', '.bval'),
                             ('bvec', '.bvec'),
                             ('TE', '.TE'),
                             ('Delta', '.Delta'),
                             ('delta', '.delta'),
                             ('prtcl', '.prtcl'),
                             ('dwi', '.nii'), ('dwi', '.nii.gz'),
                             ('mask', '_mask.nii'), ('mask', '_mask.nii.gz')]

        default_mask = None
        if glob.glob(os.path.join(self._root_dir, 'mask.nii*')):
            default_mask = glob.glob(os.path.join(self._root_dir, 'mask.nii*'))[0]

        for basename in basenames:
            info = {}
            for info_key, ext in files_to_look_for:
                if basename + ext in files:
                    info.update({info_key: os.path.join(self._root_dir, basename + ext)})

            if 'mask' not in info:
                info.update({'mask': default_mask})

            if 'dwi' in info and (('bval' in info and 'bvec' in info) or 'prtcl' in info):
                protocol = self._get_protocol(info)
                subjects.append(BatchSubjectInfo(basename, info['dwi'], protocol, info))
        return subjects

    def __str__(self):
        return meta_info['title']
