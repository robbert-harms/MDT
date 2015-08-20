import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
from mdt.utils import get_slice_in_dimension
from matplotlib.gridspec import GridSpec
import numpy as np


__author__ = 'Robbert Harms'
__date__ = "2014-02-14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MapsVisualizer(object):

    def __init__(self, volumes_dict):
        self._volumes_dict = volumes_dict

        self.placement_ratio = 3.0
        self.dimension = 0
        self.slice_ind = 0
        self.volume_ind = 0
        self.show_sliders = True
        self.show_slider_volume_ind = self._get_max_4d_length() > 1
        self.general_plot_options = {}
        self.map_plot_options = {}
        self._volumes_shape = self._get_volumes_shape()
        self._figure = plt.figure(figsize=(18, 16))
        self.maps_to_show = sorted(self._volumes_dict.keys())
        self.names = {}
        self.general_plot_options = {}
        self.font_size = None
        self._image_subplots = {}
        self._minmax_vals = self._load_min_max_vals()
        self._dimension_slider = None
        self._index_slider = None
        self._volume_slider = None
        self._updating_sliders = False
        self._colorbar_subplots = {}

    def show(self, dimension=None, slice_ind=None, volume_ind=None, names=None, maps_to_show=None,
             general_plot_options=None, map_plot_options=None, to_file=None, block=True, maximize=False,
             window_title=None):
        """Show the data contained in this visualizer using the specifics in this function call.

        Args:
            dimension (int):
                The dimension to display
            slice_ind (int):
                The slice (in that dimension) to display
            volume_index (dict):
                The volume to display initially.
            names (dict):
                A list of names for the different maps. Use as {map_name: display_name} that is,
                 the key is the name of the map in the volumes dictionary and the display name is the string that will
                 be used as title for that map.
            maps_to_show (list):
                A list of maps to show. The items in this list must correspond to the keys in the volumes dictionary.
            general_plot_options (dict):
                A number of options for rendering the maps. These hold for all the displayed maps.
            map_plot_options (dict):
                A number of options for rendering the maps. These options should be like:
                    {map_name: {options}}. That is a set of options for that specific map. These override the
                     general plot options if present.
            to_file (string, optional, default None):
                If to_file is not None it is supposed to be a filename where the image will be saved.
                If not set to None, nothing will be displayed, the results will directly be saved.
                Already existing items will be overwritten.
            block (boolean): If we want to block after calling the plots or not. Set this to False if you
                do not want the routine to block after drawing. In doing so you manually need to block.
            maximize (boolean): if we want to display the window maximized or not
            window_title (str): the title of the window. If None, the default title is used
        """
        if dimension is not None:
            self.dimension = dimension
        else:
            self.dimension = 2

        if slice_ind is not None:
            self.slice_ind = slice_ind
        else:
            self.slice_ind = int(self._volumes_shape[self.dimension] / 2.0)

        if volume_ind is not None:
            self.volume_ind = volume_ind
        if names:
            self.names = names
        if maps_to_show:
            self.maps_to_show = maps_to_show
        if general_plot_options:
            self.general_plot_options = general_plot_options
        if map_plot_options:
            self.map_plot_options = map_plot_options

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

    def set_dimension(self, val):
        val = round(val)
        if not self._updating_sliders:
            self._updating_sliders = True
            self.dimension = int(round(val))

            if self.slice_ind >= self._volumes_shape[self.dimension]:
                self.slice_ind = self._volumes_shape[self.dimension] / 2
            self._index_slider.set_max(self._volumes_shape[self.dimension] - 1)

            self._dimension_slider.set_val(val)
            self._rerender_maps()
            self._updating_sliders = False

    def set_slice_ind(self, val):
        val = round(val)

        if not self._updating_sliders:
            self._updating_sliders = True
            self.slice_ind = val

            if self.slice_ind < 0 or self.slice_ind >= self._volumes_shape[self.dimension]:
                self.slice_ind = self._volumes_shape[self.dimension] / 2
                self._index_slider.set_max(self._volumes_shape[self.dimension] - 1)

            self._index_slider.set_val(val)
            self._rerender_maps()
            self._updating_sliders = False

    def set_volume_ind(self, val):
        val = round(val)

        if val >= self._get_max_4d_length():
            val = self._get_max_4d_length() - 1

        if not self._updating_sliders:
            self._updating_sliders = True
            self.volume_ind = val
            self._volume_slider.set_val(val)
            self._rerender_maps()
            self._updating_sliders = False

    def _setup(self):
        if self.font_size:
            matplotlib.rcParams.update({'font.size': self.font_size})

        self._rerender_maps()

        y_positions = [0.038, 0.023, 0.008]
        if not self.show_slider_volume_ind:
            y_positions = y_positions[1:]

        if self.show_sliders:
            ax = self._figure.add_axes([0.25, y_positions[0], 0.5, 0.01], axisbg='Wheat')
            self._dimension_slider = _DiscreteSlider(ax, 'Dimension', 0, 2,
                                                     valinit=self.dimension, valfmt='%i',
                                                     color='DarkSeaGreen', closedmin=True, closedmax=True)
            self._dimension_slider.on_changed(self.set_dimension)

            ax = self._figure.add_axes([0.25, y_positions[1], 0.5, 0.01], axisbg='Wheat')
            self._index_slider = _DiscreteSlider(ax, 'Slice index', 0,
                                                 self._volumes_shape[self.dimension] - 1,
                                                 valinit=self.slice_ind,
                                                 valfmt='%i', color='DarkSeaGreen', closedmin=True, closedmax=False)
            self._index_slider.on_changed(self.set_slice_ind)

            if self.show_slider_volume_ind:
                ax = self._figure.add_axes([0.25, y_positions[2], 0.5, 0.01], axisbg='Wheat')
                self._volume_slider = _DiscreteSlider(ax, 'Volume', 0,
                                                      self._get_max_4d_length() - 1,
                                                      valinit=self.volume_ind,
                                                      valfmt='%i', color='DarkSeaGreen', closedmin=True, closedmax=False)
                self._volume_slider.on_changed(self.set_volume_ind)

    def _rerender_maps(self):
        for f in self._image_subplots.values():
            self._figure.delaxes(f)
        self._image_subplots = {}

        rows, cols = self._get_row_cols()
        grid = GridSpec(rows, cols, left=0.04, right=0.96, top=0.97, bottom=0.07)

        ind = 0
        for map_name in self.maps_to_show:
            data = self._get_image(self._volumes_dict[map_name])

            minval = self._minmax_vals[map_name][0]
            maxval = self._minmax_vals[map_name][1]

            title = map_name
            if map_name in self.names:
                title = self.names[map_name]

            image_subplots = plt.subplot(grid[ind])
            plot_options = {'vmin': minval, 'vmax': maxval}
            plot_options.update(self.general_plot_options)
            if map_name in self.map_plot_options:
                plot_options.update(self.map_plot_options[map_name])

            try:
                vf = image_subplots.imshow(data, **plot_options)
                plt.title(title)
                self._image_subplots.update({map_name: image_subplots})

                cbar = plt.colorbar(vf)
                self._colorbar_subplots.update({map_name: cbar})

            except TypeError:
                pass

            ind += 1

        self._figure.canvas.draw()

    def _get_image(self, data):
        """Get the 2d image to display for the given data.

        This will use the current knowledge about the dimensions, slice_ind and volume_ind to get the correct image
        to show.

        After getting the right image it will apply the transformations in _apply_transformations to position the
        image in a nice way.
        """
        data = get_slice_in_dimension(data, self.dimension, self.slice_ind)
        if len(data.shape) > 2:
            if data.shape[2] > self.volume_ind:
                data = np.squeeze(data[:, :, self.volume_ind])
            else:
                data = np.squeeze(data[:, :, 0])
        data = self._apply_transformations(data)
        return data

    def _apply_transformations(self, data):
        data = np.transpose(data)
        data = np.flipud(data)
        return data

    def _get_row_cols(self):
        length = len(self.maps_to_show)

        defaults = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (2, 3), (2, 3))
        if length < len(defaults):
            return defaults[length - 1]
        else:
            cols = math.ceil(length / self.placement_ratio)
            rows = math.ceil(float(length) / cols)
            rows = int(rows)
            cols = int(cols)

        return rows, cols

    def _get_volumes_shape(self):
        return self._volumes_dict[self._volumes_dict.keys()[0]].shape

    def _load_min_max_vals(self):
        d = {}
        for key, value in self._volumes_dict.items():
            try:
                d.update({key: (value.min(), value.max())})
            except TypeError:
                d.update({key: (0, 1)})
        return d

    def _get_max_4d_length(self):
        """Get the maximum volume index in the volumes."""
        l = [v.shape[3] for v in self._volumes_dict.values() if len(v.shape) > 3]
        if not l:
            return 0
        return max(l)


class SampleVisualizer(object):

    def __init__(self, voxels):
        self._voxels = voxels
        self.voxel_ind = 0
        self.maps_to_show = sorted(self._voxels.keys())
        self.names = {}
        self._figure = plt.figure(figsize=(18, 16))
        self.show_sliders = True
        self._max_voxel_ind = 0
        self._updating_sliders = False
        self._voxel_slider = None

    def show(self, voxel_ind=0, names=None, maps_to_show=None, to_file=None, block=True, maximize=False,
             window_title=None):
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
            window_title (str): the title of the window. If None, the default title is used

        """
        if names:
            self.names = names
        if maps_to_show:
            self.maps_to_show = maps_to_show
        self.voxel_ind = voxel_ind
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
        grid = GridSpec(len(self.maps_to_show)*2, 1, left=0.04, right=0.96, top=0.97, bottom=0.04, hspace=0.4)

        i = 0
        for map_name in self.maps_to_show:
            samples = self._voxels[map_name]

            title = map_name
            if map_name in self.names:
                title = self.names[map_name]

            hist_plot = plt.subplot(grid[i])
            hist_plot.hist(samples[self.voxel_ind, :], 10)
            plt.title(title)

            chain_plot = plt.subplot(grid[i+1])
            chain_plot.plot(samples[self.voxel_ind, :])

            i += 2


class _DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""

    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
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
        for cid, func in self.observers.iteritems():
            func(discrete_val)
