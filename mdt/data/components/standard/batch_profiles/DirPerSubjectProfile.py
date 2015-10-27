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

        protocol_extra_cols_search = ['TE', 'Delta', 'delta']

        for subject_id in dirs:
            extra_protocol_cols = {}

            niftis = glob.glob(os.path.join(self._root_dir, subject_id, '*.nii*'))
            dwis = list(filter(lambda v: '_mask' not in v, niftis))
            masks = list(filter(lambda v: '_mask' in v, niftis))
            protocols = glob.glob(os.path.join(self._root_dir, subject_id, '*prtcl'))
            bvals = glob.glob(os.path.join(self._root_dir, subject_id, '*bval*'))
            bvecs = glob.glob(os.path.join(self._root_dir, subject_id, '*bvec*'))

            if dwis:
                dwi_fname = dwis[0]
                mask_fname = masks[0] if masks else None
                protocol_fname = protocols[0] if protocols else None
                bval_fname = bvals[0] if bvals else None
                bvec_fname = bvecs[0] if bvecs else None

                for key in protocol_extra_cols_search:
                    items = glob.glob(os.path.join(self._root_dir, subject_id, '*' + key))
                    if items:
                        extra_protocol_cols.update({key: items[0]})

                if dwi_fname and (protocol_fname or (bval_fname and bvec_fname)):
                    protocol_loader = BatchFitProtocolLoader(
                        prtcl_fname=protocol_fname, bvec_fname=bvec_fname,
                        bval_fname=bval_fname, extra_cols_from_file=extra_protocol_cols)

                    output_dir = os.path.join(self._root_dir, subject_id, 'output')

                    subjects.append(SimpleSubjectInfo(subject_id, dwi_fname, protocol_loader, mask_fname, output_dir))

        return subjects

    def __str__(self):
        return meta_info['title']