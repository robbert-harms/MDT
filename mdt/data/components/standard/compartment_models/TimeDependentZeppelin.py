from mdt import CompartmentTemplate

__author__ = 'Robbert Harms'
__date__ = "2015-06-21"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class TimeDependentZeppelin(CompartmentTemplate):
    """Implements a Zeppelin (cylindrical symmetric Tensor) with time dependence in the perpendicular diffusivity.
    The perpendicular diffusivity is calculated as:

    .. math::

        D_{h, \perp} = D_{h,\infty} + A \frac{\ln(\Delta/\delta) + 3/2}{\Delta - \delta/3}


    For a detailed description please see equation 11 in [1].

    References:
        [1] De Santis, S., Jones D., Roebroeck A., 2016. Including diffusion time dependence in the extra-axonal space
            improves in vivo estimates of axonal diameter and density in human white matter, NeuroImage 2016.
    """
    parameters = ('g', 'b', 'd', 'd_bulk', 'theta', 'phi', 'time_dependent_characteristic_coefficient(A)',
                  'Delta', 'delta')
    dependencies = ('Zeppelin',)
    cl_code = '''
        mot_float_type dperp0 = d_bulk + A * (log(Delta/delta) + 3/2.0)/(Delta - delta/3.0);
        return Zeppelin(g, b, d, dperp0, theta, phi);
    '''
