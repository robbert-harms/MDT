import glob
import os
import mdt
from mdt.batch_utils import BatchSubjectInfo, SimpleBatchProfile

__author__ = 'Robbert Harms'
__date__ = "2015-07-13"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

meta_info = {'title': 'HCP WU-Minn',
             'description': 'The profile for the WU-Minn data from the Human Connectome project',
             'directory_layout':
'''
This assumes that you downloaded and extracted the WU-Minn data in one folder and that you now have
one folder per subject.

Example directory layout:
    /*/T1w/Diffusion/data.nii.gz
    /*/T1w/Diffusion/bvals
    /*/T1w/Diffusion/bvecs
    /*/T1w/Diffusion/nodif_brain_mask.nii.gz

Optional items (these will take precedence if present):
    /*/T1w/Diffusion/data.bval
    /*/T1w/Diffusion/data.bvec
    /*/T1w/Diffusion/data.prtcl
    /*/T1w/Diffusion/data_mask.nii(.gz)
'''}

class HCP_WUMINN_Profile(SimpleBatchProfile):

    def get_output_directory(self, subject_id):
        return os.path.join(self._root_dir, subject_id, 'T1w', 'Diffusion', 'output')

    def _get_subjects(self):
        dirs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self._root_dir, '*'))])
        subjects = []
        for d in dirs:
            info = {}

            pjoin = mdt.make_path_joiner(self._root_dir, d, 'T1w', 'Diffusion')
            if os.path.isdir(pjoin()):
                if glob.glob(pjoin('data.nii*')):
                    info['dwi'] = glob.glob(pjoin('data.nii*'))[0]

                if os.path.isfile(pjoin('data.bval')):
                    info['bval'] = pjoin('data.bval')
                elif os.path.isfile(pjoin('bvals')):
                    info['bval'] = pjoin('bvals')

                if os.path.isfile(pjoin('data.bvec')):
                    info['bvec'] = pjoin('data.bvec')
                elif os.path.isfile(pjoin('bvecs')):
                    info['bvec'] = pjoin('bvecs')

                if os.path.isfile(pjoin('data.prtcl')):
                    info['prtcl'] = pjoin('data.prtcl')

                if glob.glob(pjoin('data_mask.nii*')):
                    info['mask'] = glob.glob(pjoin('data_mask.nii*'))[0]
                elif glob.glob(pjoin('nodif_brain_mask.nii*')):
                    info['mask'] = glob.glob(pjoin('nodif_brain_mask.nii*'))[0]

            if 'dwi' in info and (('bval' in info and 'bvec' in info) or 'prtcl' in info):
                protocol = self._get_protocol(info)
                subjects.append(BatchSubjectInfo(d, info['dwi'], protocol, info))
        return subjects

    def _get_protocol(self, found_items):
        protocol = super(HCP_WUMINN_Profile, self)._get_protocol(found_items)
        for col, val in {'TE': 0.0895}.items():
            protocol.add_column(col, val)
        protocol.add_estimated_protocol_params(maxG=0.1)
        return protocol

    def __str__(self):
        return meta_info['title']