from mdt import BatchProfileTemplate

__author__ = 'Robbert Harms'
__date__ = '2018-03-18'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class DirPerSubject(BatchProfileTemplate):
    """General layout for batch fitting with a folder per subject.

    Every subject has its own directory. For every type of file (protocol, bvec, bval, TE, Delta, delta, DWI, mask)
    we use the first one found.

    The pattern matching of the DWI data takes into account filtering the niftis that were already matched by the
    mask and gradient deviations file.

    You can provide the noise standard deviation to use using a noise_std file containing a single float.

    The optional items TE, Delta and delta provide extra information about the protocol.
    They should either contain exactly 1 value (for all protocol lines), or a value per protocol line.

    Example directory layout:
        /a/data.nii(.gz)
        /a/bval
        /a/bvec

        /b/...
        /c/...

    Optional items:
        /*/prtcl
        /*/*.prtcl

        /*/TE (in seconds)
        /*/delta (in seconds)
        /*/Delta (in seconds) (it is also possible to use a filename /*/big_delta for case insensitive filesystems)

        /*/*_mask.nii(.gz)

        /*/noise_std.{txt,nii,nii.gz}
    """
