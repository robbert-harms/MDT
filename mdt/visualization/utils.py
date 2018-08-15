import numpy as np
from matplotlib.ticker import LinearLocator

__author__ = 'Robbert Harms'
__date__ = "2014-02-14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MyColourBarTickLocator(LinearLocator):

    def __init__(self, min_val, max_val, round_precision=3, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.round_precision = round_precision

    def __call__(self):
        locations = LinearLocator.__call__(self)

        new_locations = []
        for location in locations:
            if np.absolute(location) < 0.01:
                new_locations.append(float("{:.1e}".format(location)))
            else:
                new_locations.append(np.round(location, self.round_precision))

        if np.isclose(new_locations[-1], self.max_val) or new_locations[-1] >= self.max_val:
            new_locations[-1] = self.max_val

        if new_locations[0] <= self.min_val:
            new_locations[0] = self.min_val

        return new_locations
