import os
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm
from mdt.utils import get_slice_in_dimension
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.mlab as mlab


__author__ = 'Robbert Harms'
__date__ = "2014-02-14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MapsVisualizer(object):

    def __init__(self, volumes_dict, figure):
        self._volumes_dict = volumes_dict
        self._figure = figure

    def render(self, plot_config):
        """Render all the maps to the figure. This is for use in GUI embedded situations."""
        Renderer(self._volumes_dict, self._figure, plot_config).render()

    def to_file(self, file_name, plot_config):
        """Renders the figures to the given filename."""
        Renderer(self._volumes_dict, self._figure, plot_config).render()

        if not os.path.isdir(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)

    def show(self, plot_config, block=True, maximize=False, window_title=None):
        """Show the data contained in this visualizer using the specifics in this function call.

        Args:
            plot_config (PlotConfig): the plot configuration
            block (boolean): If we want to block after calling the plots or not. Set this to False if you
                do not want the routine to block after drawing. In doing so you manually need to block.
            maximize (boolean): if we want to display the window maximized or not
            window_title (str): the title of the window. If None, the default title is used
        """
        Renderer(self._volumes_dict, self._figure, plot_config).render()

        if maximize:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()

        if window_title:
            mng = plt.get_current_fig_manager()
            mng.canvas.set_window_title(window_title)

        if block:
            plt.show(True)


class PlotConfig(object):

    def __init__(self, dimension=0, slice_index=0, volume_index=0, rotate=90, colormap='hot', maps_to_show=(),
                 font_size=14, grid_layout=None, colorbar_nmr_ticks=10, show_axis=True, zoom=None,
                 map_plot_options=None):
        """Container for all plot related settings."""
        self.dimension = dimension
        self.slice_index = slice_index
        self.volume_index = volume_index
        self.rotate = rotate
        self.colormap = colormap
        self.maps_to_show = maps_to_show
        self.font_size = font_size
        self.colorbar_nmr_ticks = colorbar_nmr_ticks
        self.grid_layout = grid_layout or AutoGridLayout()
        self.show_axis = show_axis
        self.zoom = zoom
        self.map_plot_options = map_plot_options


class Renderer(object):

    def __init__(self, volumes_dict, figure, plot_config):
        """Create a new renderer for the given volumes on the given figure using the given configuration.

        Args:
            volumes_dict (dict): the dictionary with the maps to show
            figure (Figure): the matplotlib figure to draw on
            plot_config (PlotConfig): the plot configuration
        """
        self._volumes_dict = volumes_dict
        self._figure = figure
        self._plot_config = plot_config

    def render(self):
        """Render the maps"""
        for ind, map_name in enumerate(self._plot_config.maps_to_show):
            axis = self._plot_config.grid_layout.get_axis(self._figure, ind, len(self._plot_config.maps_to_show))
            self._render_map(map_name, axis)

    def _render_map(self, map_name, axis):
        axis.set_title(self._get_title(map_name))
        axis.axis('on' if self._plot_config.show_axis else 'off')

        image_data = ImageData(self._get_image(map_name)) \
            .rotate(self._plot_config.rotate) \
            .clip(self._get_map_specific_option(map_name, 'clipping', {})) \
            .zoom(self._plot_config.zoom)

        plot_options = self._get_map_plot_options(map_name)
        vf = axis.imshow(image_data.data, **plot_options)

        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)

        self._add_colorbar(map_name, colorbar_axis, vf)
        self._apply_font_size(axis, colorbar_axis)

    def _apply_font_size(self, image_axis, colorbar_axis):
        items = [image_axis.xaxis.label, image_axis.yaxis.label]
        items.extend(image_axis.get_xticklabels())
        items.extend(image_axis.get_yticklabels())

        for item in items:
            item.set_fontsize(self._plot_config.font_size - 2)

        image_axis.title.set_fontsize(self._plot_config.font_size)
        colorbar_axis.tick_params(labelsize=self._plot_config.font_size - 2)
        colorbar_axis.yaxis.offsetText.set(size=self._plot_config.font_size - 3)

    def _add_colorbar(self, map_name, axis, image_figure):
        cbar = plt.colorbar(image_figure, cax=axis, ticks=self._get_tick_locator(map_name))
        cbar.formatter.set_powerlimits((-3, 4))
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()

        if cbar.ax.get_yticklabels():
            cbar.ax.get_yticklabels()[-1].set_verticalalignment('top')
        return cbar

    def _get_map_specific_option(self, map_name, option, default):
        if map_name in self._plot_config.map_plot_options:
            map_options = self._plot_config.map_plot_options[map_name]
            value = map_options.get(option, default)
            if value:
                return value
        return default

    def _get_title(self, map_name):
        return self._get_map_specific_option(map_name, 'title', map_name)

    def _get_map_plot_options(self, map_name):
        output_dict = {'vmin': self._volumes_dict[map_name].min(),
                       'vmax': self._volumes_dict[map_name].max(),
                       'cmap': self._get_map_specific_option(map_name, 'colormap', self._plot_config.colormap)}

        scale = self._get_map_specific_option(map_name, 'scale', {'max': None, 'min': None})
        if scale.get('max') is not None:
            output_dict['vmax'] = scale['max']
        if scale.get('min') is not None:
            output_dict['vmin'] = scale['min']

        return output_dict

    def _get_image(self, map_name):
        """Get the 2d image to display for the given data."""
        data = self._volumes_dict[map_name]
        dimension = self._plot_config.dimension
        slice_index = self._plot_config.slice_index
        volume_index = self._plot_config.volume_index

        data = get_slice_in_dimension(data, dimension, slice_index)
        if len(data.shape) > 2:
            data = np.squeeze(data[:, :, volume_index])
        return data

    def _get_tick_locator(self, map_name):
        min_val, max_val = self._volumes_dict[map_name].min(), self._volumes_dict[map_name].max()
        return MyColourBarTickLocator(min_val, max_val, numticks=self._plot_config.colorbar_nmr_ticks)


class ImageData(object):

    def __init__(self, data):
        """Container for the displayed image data. Has functionality to change the image data."""
        self.data = data

    def rotate(self, factor):
        """Apply rotation and return new a new ImageData object.

        Args:
            factor (int): the angle to rotate by, must be a multiple of 90.
        """
        if factor:
            return ImageData(np.rot90(self.data, factor // 90))
        return self

    def clip(self, clipping):
        """Apply clipping and return new a new ImageData object.

        This function applies basic checks on the clipping dict before clipping.

        Args:
            clipping (dict): the clipping information. Keys: 'min' and 'max'.
        """
        if clipping:
            clipping_min = clipping.get('min', None)
            if clipping_min is None:
                clipping_min = self.data.min()

            clipping_max = clipping.get('max', None)
            if clipping_max is None:
                clipping_max = self.data.max()

            if clipping_min or clipping_max:
                return ImageData(np.clip(self.data, clipping_min, clipping_max))
        return self

    def zoom(self, zoom):
        """Apply zoom and return new a new ImageData object.

        This function applies basic checks on the zoom dict before zooming.

        Args:
           zoom (dict): the zoom information. Keys: 'x_0', 'x_1', 'y_0', 'y_1'
        """
        if zoom:
            correct = all(map(lambda e: e in zoom and zoom[e] is not None and zoom[e] >= 0,
                              ('x_0', 'x_1', 'y_0', 'y_1'))) \
                      and zoom['x_0'] < self.data.shape[1] and zoom['x_1'] < self.data.shape[1] \
                      and zoom['y_0'] < self.data.shape[0] and zoom['y_1'] < self.data.shape[0] \
                      and zoom['x_0'] < zoom['x_1'] and zoom['y_0'] < zoom['y_1']
            if correct:
                return ImageData(self.data[zoom['y_0']:zoom['y_1'], zoom['x_0']:zoom['x_1']])
        return self


class SampleVisualizer(object):

    def __init__(self, voxels):
        self._voxels = voxels
        self.voxel_ind = 0
        self.maps_to_show = sorted(self._voxels.keys())
        self.names = {}
        self._figure = None
        self.show_sliders = True
        self._max_voxel_ind = 0
        self._updating_sliders = False
        self._voxel_slider = None
        self._show_trace = True
        self._nmr_bins = 30
        self._show_slider = True
        self._fit_gaussian = True

    def show(self, voxel_ind=0, names=None, maps_to_show=None, to_file=None, block=True, maximize=False,
             show_trace=True, nmr_bins=20, window_title=None, show_sliders=True, fit_gaussian=True,
             figure_options=None):
        """Show the samples per voxel.
        Args:
            voxel_ind (int): the voxel to show the samples from.
            names (dict):
                A list of names for the different maps. Use as {map_name: display_name} that is,
                 the key is the name of the map in the volumes dictionary and the display name is the string that will
                 be used as title for that map.
            maps_to_show (list):
                A list of maps to show. The items in this list must correspond to the keys in the volumes dictionary.
            to_file (string, optional, default None):
                If to_file is not None it is supposed to be a filename where the image will be saved.
                If not set to None, nothing will be displayed, the results will directly be saved.
                Already existing items will be overwritten.
            block (boolean): If we want to block after calling the plots or not. Set this to False if you
                do not want the routine to block after drawing. In doing so you manually need to block.
            maximize (boolean): if we want to display the window maximized or not
            show_trace (boolean): if we show the trace of each map or not
            nmr_bins (dict or int): either a single value or one per map name
            show_sliders (boolean): if we show the slider or not
            fit_gaussian (boolean): if we fit and show a normal distribution (Gaussian) to the histogram or not
            window_title (str): the title of the window. If None, the default title is used
            figure_options (dict) options for the figure
        """
        figure_options = figure_options or {'figsize': (18, 16)}
        self._figure = plt.figure(**figure_options)

        if names:
            self.names = names
        if maps_to_show:
            self.maps_to_show = maps_to_show
        self.voxel_ind = voxel_ind
        self._nmr_bins = nmr_bins or self._nmr_bins
        self._show_trace = show_trace
        self.show_sliders = show_sliders
        self._fit_gaussian = fit_gaussian

        self._setup()

        if maximize:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()

        if window_title:
            mng = plt.get_current_fig_manager()
            mng.canvas.set_window_title(window_title)

        if to_file:
            plt.savefig(to_file)
            plt.close()
        else:
            plt.draw()
            if block:
                plt.show(True)

    def set_voxel(self, voxel_ind):
        voxel_ind = round(voxel_ind)
        if not self._updating_sliders:
            self._updating_sliders = True
            self.voxel_ind = int(round(voxel_ind))

            self._voxel_slider.set_val(voxel_ind)
            self._rerender()
            self._voxel_slider.set_val(voxel_ind)
            self._updating_sliders = False

    def _setup(self):
        self._rerender()

        self._max_voxel_ind = max([self._voxels[map_name].shape[0] for map_name in self.maps_to_show])

        y_positions = [0.008]

        if self.show_sliders:
            ax = self._figure.add_axes([0.25, y_positions[0], 0.5, 0.01], axisbg='Wheat')
            self._voxel_slider = _DiscreteSlider(ax, 'Voxel', 0, self._max_voxel_ind - 1,
                                                     valinit=self.voxel_ind, valfmt='%i',
                                                     color='DarkSeaGreen', closedmin=True, closedmax=True)
            self._voxel_slider.on_changed(self.set_voxel)

    def _rerender(self):
        nmr_maps = len(self.maps_to_show)
        if self._show_trace:
            nmr_maps *= 2

        grid = GridSpec(nmr_maps, 1, left=0.04, right=0.96, top=0.97, bottom=0.04, hspace=0.4)

        i = 0
        for map_name in self.maps_to_show:
            samples = self._voxels[map_name]

            title = map_name
            if map_name in self.names:
                title = self.names[map_name]

            if isinstance(self._nmr_bins, dict) and map_name in self._nmr_bins:
                nmr_bins = self._nmr_bins[map_name]
            else:
                nmr_bins = self._nmr_bins

            hist_plot = plt.subplot(grid[i])
            n, bins, patches = hist_plot.hist(samples[self.voxel_ind, :], nmr_bins, normed=True)
            plt.title(title)
            i += 1

            if self._fit_gaussian:
                mu, sigma = norm.fit(samples[self.voxel_ind, :])
                bincenters = 0.5*(bins[1:] + bins[:-1])
                y = mlab.normpdf(bincenters, mu, sigma)
                hist_plot.plot(bincenters, y, 'r', linewidth=1)

            if self._show_trace:
                trace_plot = plt.subplot(grid[i])
                trace_plot.plot(samples[self.voxel_ind, :])
                i += 1


class _DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""

    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" and kwarg.

        Args:
            increment (float): specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 0.25)
        Slider.__init__(self, *args, **kwargs)

    def set_max(self, new_max):
        orig_val = self.val
        self.set_val(self.valmin)

        self.valmax = new_max
        self.ax.set_xlim((self.valmin, self.valmax))

        if orig_val >= new_max:
            self.set_val((new_max + self.valmin) / 2.0)
        else:
            self.set_val(orig_val)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon:
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.items():
            func(discrete_val)


class MyColourBarTickLocator(LinearLocator):

    def __init__(self, min_val,max_val, **kwargs):
        super(MyColourBarTickLocator, self).__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self):
        locations = LinearLocator.__call__(self)

        new_locations = []
        for location in locations:
            if np.absolute(location) < 0.01:
                new_locations.append(float("{:.1e}".format(location)))
            else:
                new_locations.append(np.round(location, 2))

        if np.isclose(new_locations[-1], self.max_val):
            new_locations[-1] = self.max_val

        return new_locations


class GridLayout(object):

    def __init__(self):
        self.spacings = dict(left=0.06, right=0.92, top=0.97, bottom=0.04, wspace=0.5)

    def get_axis(self, figure, index, nmr_plots):
        """Get the axis for the subplot at the given index in the data list.

        Args:
            figure (Figure): the figure to add the axis to
            index (int): the index of the subplot in the list of plots
            nmr_plots (int): the total number of plots

        Returns:
            axis: a matplotlib axis object that can be drawn on
        """


class AutoGridLayout(GridLayout):

    def get_axis(self, figure, index, nmr_plots):
        rows, cols = self._get_row_cols_square(nmr_plots)
        grid = GridSpec(rows, cols, **self.spacings)
        return figure.add_subplot(grid[index])

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


class RectangularGridLayout(GridLayout):

    def __init__(self, rows, cols):
        super(RectangularGridLayout, self).__init__()
        self.rows = rows
        self.cols = cols

    def get_axis(self, figure, index, nmr_plots):
        grid = GridSpec(self.rows, self.cols, **self.spacings)
        return figure.add_subplot(grid[index])


class LowerTriangleGridLayout(GridLayout):

    def __init__(self):
        super(LowerTriangleGridLayout, self).__init__()
        self._positions_cache = {}

    def get_axis(self, figure, index, nmr_plots):
        size, positions = self._get_size_and_position(nmr_plots)
        grid = GridSpec(size, size, **self.spacings)
        return figure.add_subplot(grid[positions[index]])

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


class SingleColumnGridLayout(GridLayout):

    def get_axis(self, figure, index, nmr_plots):
        grid = GridSpec(nmr_plots, 1, **self.spacings)
        return figure.add_subplot(grid[index])


class SingleRowGridLayout(GridLayout):

    def get_axis(self, figure, index, nmr_plots):
        grid = GridSpec(1, nmr_plots, **self.spacings)
        return figure.add_subplot(grid[index])
