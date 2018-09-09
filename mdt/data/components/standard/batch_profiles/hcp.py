from mdt import BatchProfileTemplate

__author__ = 'Robbert Harms'
__date__ = '2018-03-18'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class HCP_MGH(BatchProfileTemplate):
    """The profile for the MGH data from the Human Connectome project.

    This assumes that you downloaded and extracted the MGH data in one folder and that you now have one folder per subject.

    You can provide the noise standard deviation to use using a noise_std file containing a single float.

    Example directory layout:
        /mgh_*/diff/preproc/mri/diff_preproc.nii(.gz)
        /mgh_*/diff/preproc/bvals.txt
        /mgh_*/diff/preproc/bvecs_fsl_moco_norm.txt

    Optional items:
        /mgh_*/diff/preproc/diff_preproc.prtcl
        /mgh_*/diff/preproc/mri/diff_preproc_mask.nii(.gz)
        /mgh_*/diff/preproc/noise_std.{txt,nii,nii.gz}
    """
    subject_base_folder = '{subject_id}/diff/preproc/'

    data_fname = '{subject_base_folder}/mri/diff_preproc.nii*'
    mask_fname = '{subject_base_folder}/mri/diff_preproc_mask.nii*'

    bvec_fname = '{subject_base_folder}/bvecs_fsl_moco_norm.txt'
    bval_fname = '{subject_base_folder}/bvals.txt'
    protocol_columns = {'Delta': 21.8e-3, 'delta': 12.9e-3, 'TR': 8800e-3, 'TE': 57e-3}


class HCP_WUMINN(BatchProfileTemplate):
    """The profile for the WU-Minn data from the Human Connectome project',

    This assumes that you downloaded and extracted the WU-Minn data in one folder which gives one folder per subject.

    You can provide the noise standard deviation to use using a noise_std file containing a single float.

    Example directory layout:
        /*/T1w/Diffusion/data.nii.gz
        /*/T1w/Diffusion/bvals
        /*/T1w/Diffusion/bvecs
        /*/T1w/Diffusion/nodif_brain_mask.nii.gz
        /*/T1w/Diffusion/grad_dev.nii.gz

    Optional items:
        /*/T1w/Diffusion/data.prtcl
        /*/T1w/Diffusion/noise_std
    """
    subject_base_folder = '{subject_id}/T1w/Diffusion/'

    data_fname = '{subject_base_folder}/data.nii*'
    mask_fname = '{subject_base_folder}/nodif_brain_mask.nii*'

    gradient_deviations_fname = '{subject_base_folder}/grad_dev.nii*'

    bvec_fname = '{subject_base_folder}/bvecs'
    bval_fname = '{subject_base_folder}/bvals'
    protocol_columns = {'Delta': 43.1e-3, 'delta': 10.6e-3, 'TE': 89.5e-3, 'TR': 5520e-3}
