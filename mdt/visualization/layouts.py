import itertools

import numpy as np
from matplotlib.gridspec import GridSpec

__author__ = 'Robbert Harms'
__date__ = "2016-09-02"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GridLayout(object):

    def __init__(self):
        self.spacings = dict(left=0.06, right=0.92, top=0.97, bottom=0.04, wspace=0.5)

    def get_gridspec(self, figure, nmr_plots):
        """Get the grid layout specifier for the given figure using the given number of plots.

        Args:
            figure (Figure): the figure to add the axis to
            nmr_plots (int): the total number of plots

        Returns:
            GridLayoutSpecifier: the specifier we can ask new subplot axis from
        """

    def __eq__(self, other):
        if not isinstance(other, GridLayout):
            return NotImplemented
        return isinstance(other, type(self))

    def __iter__(self):
        return iter([])


class GridLayoutSpecifier(object):

    def __init__(self, gridspec, figure):
        """Create a grid layout specifier using the given gridspec and the given figure.

        Args:
            gridspec (GridSpec): the gridspec to use
            figure (Figure): the figure to generate subplots for
        """
        self.gridspec = gridspec
        self.figure = figure

    def get_axis(self, index):
        return self.figure.add_subplot(self.gridspec[index])


class AutoGridLayout(GridLayout):

    def get_gridspec(self, figure, nmr_plots):
        rows, cols = self._get_row_cols_square(nmr_plots)
        return GridLayoutSpecifier(GridSpec(rows, cols, **self.spacings), figure)

    def _get_row_cols_square(self, nmr_plots):
        defaults = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (2, 3), (2, 3))
        if nmr_plots < len(defaults):
            return defaults[nmr_plots - 1]
        else:
            cols = np.ceil(nmr_plots / 3.0)
            rows = np.ceil(float(nmr_plots) / cols)
            rows = int(rows)
            cols = int(cols)
        return rows, cols


class Rectangular(GridLayout):

    def __init__(self, rows=None, cols=None):
        super(Rectangular, self).__init__()
        self.rows = rows
        self.cols = cols

        if self.rows:
            self.rows = int(self.rows)
        if self.cols:
            self.cols = int(self.cols)

    def get_gridspec(self, figure, nmr_plots):
        rows = self.rows
        cols = self.cols

        if rows is None and cols is None:
            return AutoGridLayout().get_gridspec(figure, nmr_plots)

        if rows is None:
            rows = int(np.ceil(nmr_plots / cols))
        if cols is None:
            cols = int(np.ceil(nmr_plots / rows))

        return GridLayoutSpecifier(GridSpec(rows, cols, **self.spacings), figure)

    def __eq__(self, other):
        if not isinstance(other, GridLayout):
            return NotImplemented
        return isinstance(other, type(self)) and other.rows == self.rows and other.cols == self.cols

    def __iter__(self):
        yield 'rows', self.rows
        yield 'cols', self.cols


class LowerTriangle(GridLayout):

    def __init__(self):
        super(LowerTriangle, self).__init__()
        self._positions_cache = {}

    def get_gridspec(self, figure, nmr_plots):
        size, positions = self._get_size_and_position(nmr_plots)
        return GridLayoutSpecifier(GridSpec(size, size, **self.spacings), figure)

    def _get_size_and_position(self, nmr_plots):
        if nmr_plots not in self._positions_cache:
            size = self._get_lowest_triangle_length(nmr_plots)

            positions = []
            for x, y in itertools.product(range(size), range(size)):
                if x >= y:
                    positions.append(x * size + y)

            self._positions_cache.update({nmr_plots: (size, positions)})

        return self._positions_cache[nmr_plots]

    @staticmethod
    def _get_lowest_triangle_length(nmr_plots):
        for n in range(1, nmr_plots):
            if 0.5 * (n ** 2 + n) >= nmr_plots:
                return n
        return nmr_plots


class SingleColumn(GridLayout):

    def get_gridspec(self, figure, nmr_plots):
        return GridLayoutSpecifier(GridSpec(nmr_plots, 1, **self.spacings), figure)


class SingleRow(GridLayout):

    def get_gridspec(self, figure, nmr_plots):
        return GridLayoutSpecifier(GridSpec(1, nmr_plots, **self.spacings), figure)
