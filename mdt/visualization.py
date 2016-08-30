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

    def __init__(self, volumes_dict, figure=None):
        self._volumes_dict = volumes_dict
        self._figure = figure or plt.figure()

    def render(self, **kwargs):
        """Render all the maps to the figure. This is for use in GUI embedded situations."""
        self._render(**kwargs)

    def to_file(self, file_name, **kwargs):
        """Renders the figures to the given filename."""
        self._render(**kwargs)
        if not os.path.isdir(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)

    def show(self, block=True, maximize=False, window_title=None, **kwargs):
        """Show the data contained in this visualizer using the specifics in this function call.

        Args:
            block (boolean): If we want to block after calling the plots or not. Set this to False if you
                do not want the routine to block after drawing. In doing so you manually need to block.
            maximize (boolean): if we want to display the window maximized or not
            window_title (str): the title of the window. If None, the default title is used
        """
        self._render(**kwargs)

        if maximize:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()

        if window_title:
            mng = plt.get_current_fig_manager()
            mng.canvas.set_window_title(window_title)

        plt.draw()
        if block:
            plt.show(True)

    def _render(self, dimension=0, slice_index=0, volume_index=0, rotate=0, colormap='hot', maps_to_show=(),
                font_size=None, grid_layout=None, colorbar_nmr_ticks=None, show_axis=True, zoom=None,
                map_plot_options=None):
        """Render the images

        Args:
            dimension (int):
                The dimension to display
            slice_index (int):
                The slice (in that dimension) to display
            volume_index (dict):
                The volume to display initially.
            rotate (int): the degrees (counter-clockwise) by which to rotate the images before showing them.
                Should be a multiple of 90.
            colormap (str): the colormap to use for all the images
            maps_to_show (list):
                A list of maps to show. The items in this list must correspond to the keys in the volumes dictionary.
            font_size (int): the size of the fonts. This is the absolute size of the map titles, all the other
                titles are scaled relatively to this value. The default is 14.
            grid_layout (GridLayout) the grid layout to use for the rendering.
            colorbar_nmr_ticks (int): the nmr of ticks (labels) to display for each map
            show_axis (boolean): If we want to show the axii per map or not
            zoom (dict): if given a dictionary with items: 'x_0', 'x_1', 'y_0', 'y_1'
        """
        grid_layout = grid_layout or AutoGridLayout()
        font_size = font_size or 14
        map_plot_options = map_plot_options or {}

        self._render_maps(dimension, slice_index, volume_index, rotate,
                          colormap, list(reversed(list(maps_to_show))), font_size, colorbar_nmr_ticks, grid_layout,
                          show_axis, zoom, map_plot_options)

    def _render_maps(self, dimension, slice_index, volume_index, rotate, colormap, maps_to_show, font_size,
                     colorbar_nmr_ticks, grid_layout, show_axis, zoom, map_plot_options):

        for ind, map_name in enumerate(maps_to_show):
            image_subplot_axis = grid_layout.get_axis(ind, len(maps_to_show))

            data = self._get_image(self._volumes_dict[map_name], dimension, slice_index, volume_index, rotate,
                                   map_plot_options.get(map_name, {}).get('clipping', {}), zoom)

            plot_options = {'vmin': self._volumes_dict[map_name].min(),
                            'vmax': self._volumes_dict[map_name].max()}
            plot_options.update({'cmap': colormap})
            plot_options.update(self._get_map_plot_options(map_name, map_plot_options))

            vf = image_subplot_axis.imshow(data, **plot_options)

            plt.title(self._get_title(map_name, map_plot_options))
            plt.axis('on' if show_axis else 'off')

            divider = make_axes_locatable(image_subplot_axis)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(vf, cax=cax)
            self._set_colorbar_axis_ticks(cbar, colorbar_nmr_ticks)
            cbar.formatter.set_powerlimits((-3, 4))
            cbar.update_ticks()
            if cbar.ax.get_yticklabels():
                cbar.ax.get_yticklabels()[-1].set_verticalalignment('top')

            for item in ([image_subplot_axis.xaxis.label, image_subplot_axis.yaxis.label] +
                             image_subplot_axis.get_xticklabels() +
                             image_subplot_axis.get_yticklabels()):
                item.set_fontsize(font_size-2)
            image_subplot_axis.title.set_fontsize(font_size)
            cbar.ax.tick_params(labelsize=font_size-2)
            cbar.ax.yaxis.offsetText.set(size=font_size-2)

    def _get_title(self, map_name, map_plot_options):
        title = map_name
        if map_name in map_plot_options \
            and 'title' in map_plot_options[map_name] \
            and map_plot_options[map_name]['title']:
                title = map_plot_options[map_name]['title']
        return title

    def _get_map_plot_options(self, map_name, map_plot_options):
        output_dict = {}

        if map_name in map_plot_options:
            map_options = map_plot_options[map_name]

            colormap = map_options.get('colormap', None)
            if colormap:
                output_dict['cmap'] = colormap

            scale = map_options.get('scale', {'max': None, 'min': None})
            if scale.get('max') is not None:
                output_dict['vmax'] = scale['max']
            if scale.get('min') is not None:
                output_dict['vmin'] = scale['min']

        return output_dict

    def _get_image(self, data, dimension, slice_index, volume_index, rotate, clipping, zoom):
        """Get the 2d image to display for the given data."""
        data = get_slice_in_dimension(data, dimension, slice_index)
        if len(data.shape) > 2:
            data = np.squeeze(data[:, :, volume_index])
        data = np.flipud(np.transpose(data))

        if rotate:
            data = np.rot90(data, rotate // 90)

        if clipping:
            clipping_min = clipping.get('min', None)
            if clipping_min is None:
                clipping_min = data.min()

            clipping_max = clipping.get('max', None)
            if clipping_max is None:
                clipping_max = data.max()

            if clipping_min or clipping_max:
                data = np.clip(data, clipping_min, clipping_max)

        if zoom:
            if all(map(lambda e: e in zoom and zoom[e] is not None, ('x_0', 'x_1', 'y_0', 'y_1'))):
                correct = zoom['x_0'] < data.shape[0] and zoom['x_1'] < data.shape[0] \
                    and zoom['y_0'] < data.shape[1] and zoom['y_1'] < data.shape[1] \
                    and zoom['x_0'] < zoom['x_1'] and zoom['y_0'] < zoom['y_1']
                if correct:
                    data = data[zoom['y_0']:zoom['y_1'],
                                zoom['x_0']:zoom['x_1']]

        return data

    def _set_colorbar_axis_ticks(self, cbar, colorbar_nmr_ticks):
        if colorbar_nmr_ticks is not None:
            try:
                ticks = colorbar_nmr_ticks
                tick_locator = MyColourBarTickLocator(numticks=ticks)
                cbar.locator = tick_locator
                cbar.update_ticks()
            except TypeError:
                pass
            except ValueError:
                pass


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

    def __call__(self):
        locations = LinearLocator.__call__(self)

        new_locations = []
        for location in locations:
            if np.absolute(location) < 0.01:
                new_locations.append(float("{:.1e}".format(location)))
            else:
                new_locations.append(np.round(location, 2))
        return new_locations


class GridLayout(object):

    def __init__(self):
        self.spacings = dict(left=0.04, right=0.96, top=0.97, bottom=0.015, wspace=0.5)

    def get_axis(self, index, nmr_plots):
        """Get the axis for the subplot at the given index in the data list.

        Args:
            index (int): the index of the subplot in the list of plots
            nmr_plots (int): the total number of plots

        Returns:
            axis: a matplotlib axis object that can be drawn on
        """


class AutoGridLayout(GridLayout):

    def get_axis(self, index, nmr_plots):
        rows, cols = self._get_row_cols_square(nmr_plots)
        grid = GridSpec(rows, cols, **self.spacings)
        return plt.subplot(grid[index])

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

    def get_axis(self, index, nmr_plots):
        grid = GridSpec(self.rows, self.cols, **self.spacings)
        return plt.subplot(grid[index])


class LowerTriangleGridLayout(GridLayout):

    def __init__(self, size):
        super(LowerTriangleGridLayout, self).__init__()

        self._size = size
        self._positions = []

        for y, x in itertools.product(range(self._size), range(self._size)):
            if x >= y:
                self._positions.append(x * self._size + y)

    def get_axis(self, index, nmr_plots):
        grid = GridSpec(self._size, self._size, **self.spacings)
        return plt.subplot(grid[self._positions[index]])


class SingleColumnGridLayout(GridLayout):

    def get_axis(self, index, nmr_plots):
        grid = GridSpec(nmr_plots, 1, **self.spacings)
        return plt.subplot(grid[index])


class SingleRowGridLayout(GridLayout):

    def get_axis(self, index, nmr_plots):
        grid = GridSpec(1, nmr_plots, **self.spacings)
        return plt.subplot(grid[index])
