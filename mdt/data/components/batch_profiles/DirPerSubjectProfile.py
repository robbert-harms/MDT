import glob
import os
from mdt.batch_utils import BatchSubjectInfo, SimpleBatchProfile

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

Example directory layout:
    /a/*.nii(.gz)
    /a/*.bval
    /a/*.bvec

    /b/...
    /c/...

Optional items:
    /*/prtcl
    /*/*.prtcl

    /*/TE (in seconds)
    /*/*.TE (in seconds)

    /*/delta (in seconds)
    /*/*.delta (in seconds)

    /*/Delta (in seconds)
    /*/*.Delta (in seconds)

    /*/*_mask.nii(.gz)

The optional items TE, Delta and delta provide extra information about the protocol.
They should either contain exactly 1 value (for all protocol lines), or a value per protocol line.

It is better to create the protocol directly by creating a .prtcl file, but adding single value files is also possible.
'''}

class DirPerSubjectProfile(SimpleBatchProfile):

    def _get_subjects(self):
        dirs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self._root_dir, '*'))])
        subjects = []

        items_to_look_for = [('bval', '*bval'), ('bvec', '*bvec'), ('TE', '*TE'), ('Delta', '*Delta'),
                             ('delta', '*delta'), ('prtcl', '*prtcl')]

        for subject_id in dirs:
            info = {}

            niftis = glob.glob(os.path.join(self._root_dir, subject_id, '*.nii*'))
            dwis = filter(lambda v: '_mask' not in v, niftis)
            masks = filter(lambda v: '_mask' in v, niftis)

            if dwis:
                info.update({'dwi': dwis[0]})

                if masks:
                    info.update({'mask': masks[0]})

                for key, pattern in items_to_look_for:
                    items = glob.glob(os.path.join(self._root_dir, subject_id, pattern))
                    if items:
                        info.update({key: items[0]})

                if 'dwi' in info and (('bval' in info and 'bvec' in info) or 'prtcl' in info):
                    protocol = self._get_protocol(info)
                    subjects.append(BatchSubjectInfo(subject_id, info['dwi'], protocol, info))

        return subjects

    def __str__(self):
        return meta_info['title']