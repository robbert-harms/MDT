import itertools
import numpy as np
from matplotlib.gridspec import GridSpec
from mdt.visualization.dict_conversion import SimpleClassConversion, IntConversion, SimpleDictConversion

__author__ = 'Robbert Harms'
__date__ = "2016-09-02"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GridLayout:

    def __init__(self, spacings=None):
        super().__init__()
        self.spacings = spacings or {'left': 0.10, 'right': 0.86,
                                     'top': 0.97, 'bottom': 0.03,
                                     'wspace': 0.40, 'hspace': 0.15}

        if self.spacings['top'] <= self.spacings['bottom']:
            raise ValueError('The top ({}) can not be smaller than the bottom ({}) in the spacings'.format(
                self.spacings['top'], self.spacings['bottom']))
        if self.spacings['left'] >= self.spacings['right']:
            raise ValueError('Left ({}) can not be larger than right ({}) in the spacings'.format(
                self.spacings['left'], self.spacings['right']))

    @classmethod
    def get_conversion_info(cls):
        return SimpleClassConversion(cls, cls._get_attribute_conversions())

    @classmethod
    def _get_attribute_conversions(cls):
        return {'spacings': SimpleDictConversion(conversion_func=float)}

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
            return False
        return isinstance(other, type(self)) and other.spacings == self.spacings

    def __ne__(self, other):
        return not self.__eq__(other)


class GridLayoutSpecifier:

    def __init__(self, gridspec, figure, positions=None):
        """Create a grid layout specifier using the given gridspec and the given figure.

        Args:
            gridspec (GridSpec): the gridspec to use
            figure (Figure): the figure to generate subplots for
            positions (:class:`list`): if given, a list with grid spec indices for every requested axis
                can be logical indices or (x, y) coordinate indices (choose one and stick with it).
        """
        self.gridspec = gridspec
        self.figure = figure
        self.positions = positions

    def get_axis(self, index):
        gridspec_ind = self.gridspec[index]
        if self.positions is not None:
            gridspec_ind = self.gridspec[self.positions[index]]
        return self.figure.add_subplot(gridspec_ind)


class AutoGridLayout(GridLayout):

    def get_gridspec(self, figure, nmr_plots):
        rows, cols = self._get_square_size(nmr_plots)
        return GridLayoutSpecifier(GridSpec(rows, cols, **self.spacings), figure)

    def _get_square_size(self, nmr_plots):
        defaults = ((1, 1), (1, 2), (2, 2), (2, 2), (2, 3), (2, 3), (2, 3))
        if nmr_plots < len(defaults):
            return defaults[nmr_plots - 1]
        else:
            cols = np.ceil(nmr_plots / 3.0)
            rows = np.ceil(float(nmr_plots) / cols)
            rows = int(rows)
            cols = int(cols)
        return rows, cols


class Rectangular(GridLayout):

    def __init__(self, rows=None, cols=None, spacings=None):
        super().__init__(spacings=spacings)
        self.rows = rows
        self.cols = cols

        if self.rows is not None:
            self.rows = int(self.rows)
            if self.rows < 1:
                raise ValueError('The number of rows ({}) can not be smaller than 1.'.format(self.rows))

        if self.cols is not None:
            self.cols = int(self.cols)
            if self.cols < 1:
                raise ValueError('The number of columns ({}) can not be smaller than 1.'.format(self.rows))

    @classmethod
    def _get_attribute_conversions(cls):
        conversions = super()._get_attribute_conversions()
        conversions.update({'rows': IntConversion(), 'cols': IntConversion()})
        return conversions

    def get_gridspec(self, figure, nmr_plots):
        rows = self.rows
        cols = self.cols

        if rows is None and cols is None:
            return AutoGridLayout(spacings=self.spacings).get_gridspec(figure, nmr_plots)

        if rows is None:
            rows = int(np.ceil(nmr_plots / cols))
        if cols is None:
            cols = int(np.ceil(nmr_plots / rows))

        if rows * cols < nmr_plots:
            cols = int(np.ceil(nmr_plots / rows))

        return GridLayoutSpecifier(GridSpec(rows, cols, **self.spacings), figure)

    def __eq__(self, other):
        if not isinstance(other, Rectangular):
            return False
        return isinstance(other, type(self)) and other.rows == self.rows and other.cols == self.cols \
               and other.spacings == self.spacings


class LowerTriangular(GridLayout):

    def __init__(self, padding=0, spacings=None):
        """Create a lower triangular plot layout.

        Args:
            padding (int): normally we will construct the lower triangle from the top, like::

                    *
                    * *
                    * * *

                If padding is enabled, we will pad as many images from the top as specified. For example, a padding of
                1, with 5 images yields::

                    * *
                    * * *

                Or a padding of 2 with 4 images::

                      *
                    * * *

            spacings (dict): the spacings around each plot
        """
        super().__init__(spacings=spacings)
        self.padding = padding or 0

    def get_gridspec(self, figure, nmr_plots):
        rows, columns, positions = self._get_size_and_position(nmr_plots)
        return GridLayoutSpecifier(GridSpec(rows, columns, **self.spacings), figure, positions=positions)

    @classmethod
    def _get_attribute_conversions(cls):
        conversions = super()._get_attribute_conversions()
        conversions.update({'padding': IntConversion()})
        return conversions

    def _get_size_and_position(self, nmr_plots):
        size = self._get_lowest_triangle_length(nmr_plots + self.padding)

        row_shift = self._get_biggest_triangle_length(self.padding)

        positions = []
        for x, y in itertools.product(range(size), range(size)):
            if x >= y:
                positions.append((x - row_shift, y))

        return size - row_shift, size, positions[self.padding:]

    def __eq__(self, other):
        if not isinstance(other, LowerTriangular):
            return False
        return isinstance(other, type(self)) and other.padding == self.padding

    @staticmethod
    def _get_lowest_triangle_length(nmr_plots):
        return int(np.ceil((-1 + np.sqrt(1 + 8 * nmr_plots)) / 2.))

    @staticmethod
    def _get_biggest_triangle_length(nmr_plots):
        return int(np.floor((-1 + np.sqrt(1 + 8 * nmr_plots)) / 2.))


class SingleColumn(GridLayout):

    def get_gridspec(self, figure, nmr_plots):
        return GridLayoutSpecifier(GridSpec(nmr_plots, 1, **self.spacings), figure)


class SingleRow(GridLayout):

    def get_gridspec(self, figure, nmr_plots):
        return GridLayoutSpecifier(GridSpec(1, nmr_plots, **self.spacings), figure)
